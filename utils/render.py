import os
import base64
import ast
import argparse
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import re
import sys
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.gp import gp_Pnt, gp_Dir
import time

from OCC.Display.OCCViewer import (
    Viewer3d,
    get_color_from_name,
    Quantity_Color
)
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Pnt2d
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeEdge2d,
    BRepBuilderAPI_MakeFace,
)
from OCC.Core.Geom import Geom_Curve, Geom_Surface
from OCC.Core.Geom2d import Geom2d_Curve
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.AIS import (
    AIS_Shape,
    AIS_TexturedShape,
)
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.Graphic3d import (
    Graphic3d_NOM_NEON_GNC,
    Graphic3d_MaterialAspect,
    Graphic3d_RM_RAYTRACING,
    Graphic3d_NameOfMaterial,
)

from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_TOC_RGB)

import uuid
from io import BytesIO


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def gen_uuid_4():
    return str(uuid.uuid4())

def extract_code(str):
    pattern = re.compile(rf"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(str)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def _parse_unmatched_pred_faces(compare_message: str) -> List[int]:
    """Extract unmatched pred face ids from compare_shapes message string."""
    if not compare_message:
        return []
    match = re.search(r"unmatched_pred_faces\s*=\s*(\[[^\]]*\])", compare_message)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: List[int] = []
    for x in parsed:
        try:
            out.append(int(x))
        except Exception:
            continue
    return sorted(set(out))


def _base64_to_pil_image(base64_str: str):
    from PIL import Image

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def _load_step_shape(step_path: str) -> TopoDS_Shape:
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file does not exist: {step_path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}, status={int(status)}")
    if reader.TransferRoots() <= 0:
        raise RuntimeError(f"Failed to transfer STEP roots: {step_path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP contains no valid shape: {step_path}")
    return shape


def _extract_face_shapes(shape: TopoDS_Shape) -> List[TopoDS_Shape]:
    faces: List[TopoDS_Shape] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        if not face.IsNull():
            faces.append(face)
        explorer.Next()
    return faces


def _run_compare_shapes_cli(
    pred_step_path: str,
    gt_step_path: str,
    tol: float,
) -> Tuple[bool, str, Optional[float], List[int], List[int]]:
    """
    Run compare_shapes in a subprocess to avoid cadquery/pythonocc symbol conflicts.
    Returns (is_equal, message, iou, unmatched_pred_faces, unmatched_gt_faces).
    """
    compare_code = (
        "import json, os, sys\n"
        "project_root = sys.argv[4]\n"
        "if project_root not in sys.path:\n"
        "    sys.path.append(project_root)\n"
        "try:\n"
        "    from mcp_server.same_brep import compare_shapes_with_unmatched_faces\n"
        "    flag, msg, iou, unmatched_pred_faces, unmatched_gt_faces = compare_shapes_with_unmatched_faces(sys.argv[1], sys.argv[2], tol=float(sys.argv[3]))\n"
        "    print(json.dumps({'flag': bool(flag), 'message': str(msg), 'iou': iou, 'unmatched_pred_faces': unmatched_pred_faces, 'unmatched_gt_faces': unmatched_gt_faces}, ensure_ascii=False))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)}, ensure_ascii=False))\n"
        "    raise\n"
    )
    cmd = [
        sys.executable,
        "-c",
        compare_code,
        pred_step_path,
        gt_step_path,
        str(float(tol)),
        PROJECT_ROOT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload: Dict[str, Any] = {}
    for line in reversed(stdout_lines):
        try:
            payload = json.loads(line)
            break
        except Exception:
            continue

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        err_msg = payload.get("error") if payload else "unknown compare subprocess error"
        raise RuntimeError(f"compare subprocess failed: {err_msg}; stderr={stderr}")

    if not payload:
        raise RuntimeError("compare subprocess returned no parseable JSON output")

    unmatched_pred_faces = payload.get("unmatched_pred_faces", [])
    unmatched_gt_faces = payload.get("unmatched_gt_faces", [])

    if not isinstance(unmatched_pred_faces, list):
        unmatched_pred_faces = []
    if not isinstance(unmatched_gt_faces, list):
        unmatched_gt_faces = []

    unmatched_pred_faces = [int(x) for x in unmatched_pred_faces if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()]
    unmatched_gt_faces = [int(x) for x in unmatched_gt_faces if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()]

    return (
        bool(payload.get("flag", False)),
        str(payload.get("message", "")),
        payload.get("iou", None),
        unmatched_pred_faces,
        unmatched_gt_faces,
    )


def render_pred_with_unmatched_faces(
    gt_step_path: str,
    pred_step_path: str,
    tol: float = 1e-3,
    azimuth: float = 30.0,
    elevation: float = 30.0,
    default_view: bool = False,
    save_path: str = "/tmp/tmp_pred_highlight.png",
    render_port: str = ":99",
    resolution: Union[int, tuple] = 512,
    edge_width: float = 2.0,
    material_style: str = "plaster",
    disable_lighting_shadows: bool = False,
) -> Dict[str, Any]:
    """
    Compare pred vs GT with compare_shapes and render pred faces.
    Unmatched pred faces are colored red, matched faces blue.
    """
    if not os.path.isfile(gt_step_path):
        raise FileNotFoundError(f"GT STEP file does not exist: {gt_step_path}")
    if not os.path.isfile(pred_step_path):
        raise FileNotFoundError(f"Pred STEP file does not exist: {pred_step_path}")

    is_equal, compare_message, iou, unmatched_pred_faces, _ = _run_compare_shapes_cli(pred_step_path, gt_step_path, tol)

    pred_shape = _load_step_shape(pred_step_path)
    pred_face_shapes = _extract_face_shapes(pred_shape)

    if not pred_face_shapes:
        raise ValueError("No faces found in pred STEP shape for rendering")

    pred_face_count = len(pred_face_shapes)
    unmatched_set = {
        int(fid)
        for fid in unmatched_pred_faces
        if isinstance(fid, int) and 0 <= int(fid) < pred_face_count
    }
    if not is_equal and not unmatched_set:
        # If compare fails without explicit ids (rare topological/system paths),
        # highlight all pred faces instead of silently showing all-blue.
        unmatched_set = set(range(pred_face_count))

    blue = [30.0 / 255.0, 50.0 / 255.0, 125.0 / 255.0]
    red = [220.0 / 255.0, 46.0 / 255.0, 46.0 / 255.0]
    face_colors: List[List[float]] = []
    for idx in range(pred_face_count):
        face_colors.append(red if idx in unmatched_set else blue)

    encoded_data, camera = render_brep(
        pred_face_shapes,
        azimuth=azimuth,
        elevation=elevation,
        default_view=default_view,
        save_path=save_path,
        brep_color=blue,
        shape_colors=face_colors,
        render_port=render_port,
        resolution=resolution,
        edge_width=edge_width,
        material_style=material_style,
        disable_lighting_shadows=disable_lighting_shadows,
    )

    return {
        "is_equal": bool(is_equal),
        "compare_message": compare_message,
        "iou": iou,
        "unmatched_pred_faces": sorted(unmatched_set),
        "pred_face_count": pred_face_count,
        "image_base64": encoded_data,
        "camera": camera,
    }


class my_Viewer3d(Viewer3d):

    @staticmethod
    def _apply_material_tuning(material_aspect, material_style: str = "plaster"):
        """Tune material response while preserving shape color as the visual base."""
        style = (material_style or "plaster").lower()

        if style == "plaster":
            # Brighter plaster with reduced diffuse and stronger highlights.
            diffuse = Quantity_Color(0.76, 0.76, 0.76, Quantity_TOC_RGB)
            ambient = Quantity_Color(0.20, 0.20, 0.20, Quantity_TOC_RGB)
            specular = Quantity_Color(0.32, 0.32, 0.32, Quantity_TOC_RGB)
            shininess = 0.58
        else:
            # Fallback to the previous balanced look.
            diffuse = Quantity_Color(0.78, 0.78, 0.78, Quantity_TOC_RGB)
            ambient = Quantity_Color(0.18, 0.18, 0.18, Quantity_TOC_RGB)
            specular = Quantity_Color(0.26, 0.26, 0.26, Quantity_TOC_RGB)
            shininess = 0.42

        if hasattr(material_aspect, "SetDiffuseColor"):
            try:
                material_aspect.SetDiffuseColor(diffuse)
            except Exception:
                pass
        if hasattr(material_aspect, "SetAmbientColor"):
            try:
                material_aspect.SetAmbientColor(ambient)
            except Exception:
                pass
        if hasattr(material_aspect, "SetSpecularColor"):
            try:
                material_aspect.SetSpecularColor(specular)
            except Exception:
                pass
        if hasattr(material_aspect, "SetShininess"):
            try:
                material_aspect.SetShininess(shininess)
            except Exception:
                pass
    
    def DisplayShape(
        self,
        shapes,
        material=None,
        texture=None,
        color=None,
        transparency=None,
        edge_width: Optional[float] = None,
        material_style: Optional[str] = "plaster",
        update=False,
    ):
        """display one or a set of displayable objects"""
        ais_shapes = []  # the list of all displayed shapes

        if issubclass(shapes.__class__, gp_Pnt):
            # if a gp_Pnt is passed, first convert to vertex
            vertex = BRepBuilderAPI_MakeVertex(shapes)
            shapes = [vertex.Shape()]
        elif isinstance(shapes, gp_Pnt2d):
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(shapes.X(), shapes.Y(), 0))
            shapes = [vertex.Shape()]
        elif isinstance(shapes, Geom_Surface):
            bounds = True
            toldegen = 1e-6
            face = BRepBuilderAPI_MakeFace()
            face.Init(shapes, bounds, toldegen)
            face.Build()
            shapes = [face.Shape()]
        elif isinstance(shapes, Geom_Curve):
            edge = BRepBuilderAPI_MakeEdge(shapes)
            shapes = [edge.Shape()]
        elif isinstance(shapes, Geom2d_Curve):
            edge2d = BRepBuilderAPI_MakeEdge2d(shapes)
            shapes = [edge2d.Shape()]
        # if only one shapes, create a list with a single shape
        if not isinstance(shapes, list):
            shapes = [shapes]
        # build AIS_Shapes list
        for shape in shapes:
            if material and texture or not material and texture:
                shape_to_display = AIS_TexturedShape(shape)
                (
                    filename,
                    toScaleU,
                    toScaleV,
                    toRepeatU,
                    toRepeatV,
                    originU,
                    originV,
                ) = texture.GetProperties()
                shape_to_display.SetTextureFileName(filename)
                shape_to_display.SetTextureMapOn()
                shape_to_display.SetTextureScale(True, toScaleU, toScaleV)
                shape_to_display.SetTextureRepeat(True, toRepeatU, toRepeatV)
                shape_to_display.SetTextureOrigin(True, originU, originV)
                shape_to_display.SetDisplayMode(3)
            elif material:
                shape_to_display = AIS_Shape(shape)
                if isinstance(material, Graphic3d_NameOfMaterial):
                    shape_to_display.SetMaterial(Graphic3d_MaterialAspect(material))
                else:
                    shape_to_display.SetMaterial(material)
            else:
                # TODO: can we use .Set to attach all TopoDS_Shapes
                if isinstance(shape,TopoDS_Shape):
                    shape_to_display = AIS_Shape(shape)
                else:
                    shape_to_display = shape
            ais_shapes.append(shape_to_display)

        # if not SOLO:
        #     # computing graphic properties is expensive
        #     # if an iterable is found, so cluster all TopoDS_Shape under
        #     # an AIS_MultipleConnectedInteractive
        #     #shape_to_display = AIS_MultipleConnectedInteractive()
        #     for ais_shp in ais_shapes:
        #         # TODO : following line crashes with oce-0.18
        #         # why ? fix ?
        #         #shape_to_display.Connect(i)
        #         self.Context.Display(ais_shp, False)
        # set the graphic properties
        if material is None:
            # Apply tuned material while keeping user color as the main appearance.
            tuned_material = Graphic3d_MaterialAspect(Graphic3d_NOM_NEON_GNC)
            self._apply_material_tuning(tuned_material, material_style)
            for shape_to_display in ais_shapes:
                shape_to_display.SetMaterial(tuned_material)
        if color:
            if isinstance(color, str):
                color = get_color_from_name(color)
            elif isinstance(color, int):
                color = Quantity_Color(color)
            elif isinstance(color, list):
                color = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
            else:
                color = color
            for shp in ais_shapes:
                self.Context.SetColor(shp, color, False)
        if transparency:
            for shape_to_display in ais_shapes:
                shape_to_display.SetTransparency(transparency)
        if edge_width is not None:
            # Let callers control visible edge/line thickness in render.
            width = max(1.0, float(edge_width))
            for shp in ais_shapes:
                self.Context.SetWidth(shp, width, False)
        # display the shapes
        for shape_to_display in ais_shapes:
            self.Context.Display(shape_to_display, False)
        if update:
            # especially this call takes up a lot of time...
            self.FitAll()
            self.Repaint()

        return ais_shapes
    
    

class my_render(my_Viewer3d):
    def __init__(self, screen_size=(640, 480), enable_lighting_shadows: bool = True):
        my_Viewer3d.__init__(self)
        self._point_light = None
        self._enable_lighting_shadows = bool(enable_lighting_shadows)
        # create the renderer
        self.Create()
        self.SetSize(screen_size[0], screen_size[1])
        self.SetModeShaded()
        self.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
        if self._enable_lighting_shadows:
            self._setup_lighting()
        else:
            self._setup_basic_lighting()
        self._configure_rendering_params(enable_lighting_shadows=self._enable_lighting_shadows)
        # Prefer ray tracing only when lighting/shadows are explicitly enabled.
        if self._enable_lighting_shadows and hasattr(self, "SetRaytracingMode"):
            try:
                self.SetRaytracingMode(4)
            except TypeError:
                self.SetRaytracingMode()
            except Exception:
                pass
        self.capture_number = 0

    def _configure_rendering_params(self, enable_lighting_shadows: bool = True):
        """Configure explicit shadow/ray-tracing behavior for predictable lighting."""
        try:
            if not hasattr(self, "View") or not hasattr(self.View, "ChangeRenderingParams"):
                return
            params = self.View.ChangeRenderingParams()

            if hasattr(params, "Method") and enable_lighting_shadows:
                params.Method = Graphic3d_RM_RAYTRACING
            if hasattr(params, "IsShadowEnabled"):
                params.IsShadowEnabled = bool(enable_lighting_shadows)
            if hasattr(params, "IsTransparentShadowEnabled"):
                params.IsTransparentShadowEnabled = bool(enable_lighting_shadows)
            if hasattr(params, "IsReflectionEnabled"):
                params.IsReflectionEnabled = False
            if hasattr(params, "IsGlobalIlluminationEnabled"):
                params.IsGlobalIlluminationEnabled = False
            if hasattr(params, "RaytracingDepth") and enable_lighting_shadows:
                params.RaytracingDepth = 6
            if hasattr(params, "ShadowMapResolution") and enable_lighting_shadows:
                params.ShadowMapResolution = 2048
            if hasattr(params, "ShadowMapBias") and enable_lighting_shadows:
                params.ShadowMapBias = 0.002
            if hasattr(params, "NbMsaaSamples"):
                params.NbMsaaSamples = 4
        except Exception:
            pass

    def _setup_basic_lighting(self):
        """Enable default viewer lights without custom key light/shadows."""
        viewer = self.Viewer if hasattr(self, "Viewer") else None
        try:
            if viewer is not None and hasattr(viewer, "SetDefaultLights"):
                viewer.SetDefaultLights()
                if hasattr(viewer, "SetLightOn"):
                    viewer.SetLightOn()
            elif hasattr(self, "SetDefaultLights"):
                self.SetDefaultLights()
                if hasattr(self, "SetLightOn"):
                    self.SetLightOn()
        except Exception:
            pass
        self._point_light = None

    def _setup_lighting(self):
        """Use a single point light; fallback to default lights if unavailable."""
        viewer = self.Viewer if hasattr(self, "Viewer") else None

        # Force a single-light setup by turning off existing/default lights first.
        try:
            if viewer is not None and hasattr(viewer, "SetLightOff"):
                viewer.SetLightOff()
            elif hasattr(self, "SetLightOff"):
                self.SetLightOff()
        except Exception:
            pass

        # Main point light for clearer shadow direction.
        try:
            from OCC.Core.V3d import V3d_PositionalLight

            point = gp_Pnt(2.0, -1.2, 3.5)
            light_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)

            light = None
            for ctor in (
                lambda: V3d_PositionalLight(point, light_color),
                lambda: V3d_PositionalLight(point),
                lambda: V3d_PositionalLight(),
            ):
                try:
                    light = ctor()
                    break
                except Exception:
                    continue

            if light is None:
                return

            if hasattr(light, "SetPosition"):
                try:
                    light.SetPosition(point)
                except Exception:
                    pass
            if hasattr(light, "SetColor"):
                try:
                    light.SetColor(light_color)
                except Exception:
                    pass
            if hasattr(light, "SetIntensity"):
                try:
                    light.SetIntensity(1.90)
                except Exception:
                    pass

            if viewer is None:
                return

            if hasattr(viewer, "AddLight"):
                try:
                    viewer.AddLight(light)
                    self._point_light = light
                except Exception:
                    pass
            if hasattr(viewer, "SetLightOn"):
                try:
                    viewer.SetLightOn(light)
                except TypeError:
                    viewer.SetLightOn()
                except Exception:
                    pass
        except Exception:
            # Fallback: if single point light is unavailable, restore default rig.
            try:
                if viewer is not None and hasattr(viewer, "SetDefaultLights"):
                    viewer.SetDefaultLights()
                    if hasattr(viewer, "SetLightOn"):
                        viewer.SetLightOn()
                elif hasattr(self, "SetDefaultLights"):
                    self.SetDefaultLights()
                    if hasattr(self, "SetLightOn"):
                        self.SetLightOn()
            except Exception:
                pass

    def _update_point_light_from_shapes(self, shapes):
        """Place the point light relative to model size for consistent shadows."""
        if self._point_light is None:
            return

        shape_list = shapes if isinstance(shapes, list) else [shapes]
        bbox = Bnd_Box()
        has_shape = False

        for shp in shape_list:
            if isinstance(shp, TopoDS_Shape):
                try:
                    brepbndlib.Add(shp, bbox, True)
                except TypeError:
                    brepbndlib.Add(shp, bbox)
                has_shape = True

        if not has_shape or bbox.IsVoid():
            return

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        diag = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-3)

        # Keep a stable key-light angle while scaling with model extent.
        light_pos = gp_Pnt(cx + 0.95 * diag, cy - 0.55 * diag, cz + 1.35 * diag)

        try:
            self._point_light.SetPosition(light_pos)
        except Exception:
            pass

        if hasattr(self._point_light, "SetDisplayPosition"):
            try:
                self._point_light.SetDisplayPosition(light_pos)
            except Exception:
                pass

    def DisplayShape(
        self,
        shapes,
        material=None,
        texture=None,
        color=None,
        transparency=None,
        edge_width: Optional[float] = None,
        material_style: Optional[str] = "plaster",
        update=True,
        dump_image=True,
        dump_image_path=None,
        dump_image_filename=None,
        update_light: bool = True,
    ):
        # Re-anchor light to model bbox to avoid scale-dependent shadow artifacts.
        if update_light:
            self._update_point_light_from_shapes(shapes)
        # call the "original" DisplayShape method
        r = super(my_render, self).DisplayShape(
            shapes,
            material,
            texture,
            color,
            transparency,
            edge_width,
            material_style,
            update,
        )  # always update
        if dump_image or (
            os.getenv("PYTHONOCC_OFFSCREEN_RENDERER_DUMP_IMAGE") == "1"
        ):  # dump to jpeg file
            timestamp = ("%f" % time.time()).split(".")[0]

            if os.getenv("PYTHONOCC_OFFSCREEN_RENDERER_DUMP_IMAGE_PATH"):
                path = os.getenv("PYTHONOCC_OFFSCREEN_RENDERER_DUMP_IMAGE_PATH")
                if not os.path.isdir(path):
                    raise IOError(f"{path} is not a valid path")
            elif dump_image_path is not None:
                if not os.path.isdir(dump_image_path):
                    raise IOError(f"{dump_image_path} is not a valid path")
                path = dump_image_path
            else:
                path = os.getcwd()
            if dump_image_filename is None:
                self.capture_number += 1
                image_filename = "capture-%i-%s.jpeg" % (
                    self.capture_number,
                    timestamp.replace(" ", "-"),
                )
                image_full_name = os.path.join(path, image_filename)
            else:
                image_full_name = os.path.join(path, dump_image_filename)
            self.View.Dump(image_full_name)
            if not os.path.isfile(image_full_name):
                raise IOError("OffscreenRenderer failed to render image to file")
            print(f"OffscreenRenderer content dumped to {image_full_name}")
        return r


def render_brep(
        shape,
        azimuth: Optional[float] = 0, 
        elevation:  Optional[float] = 0, 
        default_view:  Optional[bool] = True, 
        camera = None,
        save_path: Optional[str] = "/tmp/tmp.png",
        brep_color: Optional[Union[str, List[str]]] = "blue",
        render_port: Optional[str] = ":99",
        resolution: Optional[int | tuple] = 512,
        edge_width: Optional[float] = 2.0,
        material_style: Optional[str] = "plaster",
        shape_colors: Optional[List[Union[str, List[float]]]] = None,
        disable_lighting_shadows: bool = False,
        ):
    """
    Render B-rep shape to image and return base64 encoding with camera.
    
    Uses OCCT OffscreenRenderer for off-screen 3D rendering with customizable
    view, colors, and resolution.
    
    Parameters
    ----------
    shape : TopoDS_Shape
        OCCT B-rep shape to render
        
    azimuth : float, optional
        Camera azimuth angle in degrees (counter-clockwise from X-axis).
        Only used when default_view=False. Default: 0.0

    elevation : float, optional
        Camera elevation angle in degrees (up from XY plane).
        Only used when default_view=False. Default: 0.0
        
    default_view : bool, optional
        Use default isometric view if True, otherwise use custom azimuth/elevation.
        Default: True
        
    save_path : str, optional
        Temporary path for rendered PNG image. Default: "/tmp/tmp.png"
        
    brep_color : str or list, optional
        Render color specification:
        - "blue": preset blue (normalized RGB [30/255, 50/255, 125/255])
        - [R,G,B]: custom RGB (normalized or 0-255 integers)
        - None: renderer default color
        Default: "blue"
        
    render_port : str, optional
        X11 display port for off-screen rendering. Default: ":99"
        
    resolution : int or tuple, optional
        Output image resolution:
        - int: square image (N×N)
        - tuple: (width, height) pixels
        Default: 512

    edge_width : float, optional
        Thickness for rendered edges/lines. Minimum effective value is 1.0.
        Default: 2.0

    material_style : str, optional
        Material response preset:
        - "plaster": matte, soft highlights, stronger diffuse feel
        - "balanced": previous shinier neutral look
        Default: "plaster"

    shape_colors : list, optional
        Optional per-shape colors used when shape is a list.
        If provided, each color entry is applied to the corresponding shape.

    disable_lighting_shadows : bool, optional
        If True, disable custom point light and shadow/ray-tracing effects.
        Default: False
        
    Returns
    -------
    tuple (encoded_data, camera)
        encoded_data : str
            Base64-encoded PNG image data
        camera : vtkCamera
            Camera object used for rendering
        
    Examples
    --------
    >>> # Default rendering
    >>> img_data, cam = render_brep(shape)
    
    >>> # Custom view and color
    >>> img_data, cam = render_brep(
    ...     shape,
    ...     azimuth=45.0,
    ...     elevation=30.0,
    ...     default_view=False,
    ...     brep_color=[1.0, 0.0, 0.0]
    ... )
    
    Notes
    -----
    - Requires PythonOCC environment
    - Temporary file saved to /tmp (may be cleared on reboot)
    - Returns base64 for easy HTML embedding or transmission
    - start xvfb on specified port before use
    """
    os.environ['DISPLAY'] = render_port
    print(os.environ['DISPLAY'] )
    my_renderer = my_render(enable_lighting_shadows=not disable_lighting_shadows)
    my_renderer.EraseAll()

    if isinstance(resolution, int):
        my_renderer.SetSize(resolution, resolution)
    else:
        my_renderer.SetSize(resolution[0], resolution[1])
    my_renderer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    my_camera = my_renderer.camera

    def set_camera_polar(azimuth, elevation, target=[0, 0, 0], distance= 1):    
        azimuth_radius = azimuth * math.pi / 180
        elevation_radius = elevation * math.pi / 180

        eye_x = target[0] + distance * math.cos(elevation_radius) * math.cos(azimuth_radius)
        eye_y = target[1] + distance * math.cos(elevation_radius) * math.sin(azimuth_radius)
        eye_z = target[2] + distance * math.sin(elevation_radius)

        eye = gp_Pnt(eye_x, eye_y, eye_z)
        center = gp_Pnt(target[0], target[1], target[2])
        up = gp_Dir(0, 0, 1)
    
        my_camera.SetEye(eye)
        my_camera.SetCenter(center)
        my_camera.SetUp(up)

    if camera is not None:
        my_camera = camera
    elif not default_view:
        set_camera_polar(azimuth, elevation)

    if isinstance(brep_color, str):
        if brep_color == "blue":
            color = [30.0/255.0, 50.0/255.0, 125.0/255.0]
        else:
            color = None
    else:
        color = brep_color

    if isinstance(shape, list):
        # Stabilize framing/light by anchoring once on the full list, then update
        # viewport/output only on the last face draw.
        my_renderer._update_point_light_from_shapes(shape)
        last_idx = len(shape) - 1
        for idx, s in enumerate(shape):
            local_color = color
            if shape_colors is not None and idx < len(shape_colors):
                local_color = shape_colors[idx]
            my_renderer.DisplayShape(
                s,
                color=local_color,
                edge_width=edge_width,
                material_style=material_style,
                update=(idx == last_idx),
                dump_image=(idx == last_idx),
                dump_image_path='.',
                dump_image_filename=save_path,
                update_light=False,
            )
    else:
        my_renderer.DisplayShape(
                shape,
                color=color,
                edge_width=edge_width,
                material_style=material_style,
                dump_image_path='.',
                dump_image_filename=save_path,
        )

    with open(save_path, "rb") as f:
        image_data = f.read()
        encoded_data = base64.b64encode(image_data).decode('utf-8')

    os.remove(save_path)
    camera = my_renderer.camera
    return encoded_data, camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Render STEP with optional compare-based unmatched face highlighting. "
            "When --gt-step and --pred-step are provided, only pred is rendered: "
            "unmatched pred faces are red and matched faces are blue."
        )
    )
    parser.add_argument("--step", type=str, default=None, help="Single STEP path for plain rendering.")
    parser.add_argument("--gt-step", type=str, default="/data-nas/data/experiments/genyue/genyue_exp/cad_edit/autodesk_part/B7A2N74ZJBF9MZHU_1770326152.440727/src/input.step", help="GT STEP path for compare+highlight mode.")
    parser.add_argument("--pred-step", type=str, default="/data-nas/data/experiments/genyue/genyue_exp/cad_edit/autodesk_part/B7A2N74ZJBF9MZHU_1770326152.440727/tmp/trial_1781097160.step", help="Pred STEP path for compare+highlight mode.")
    parser.add_argument("--tol", type=float, default=1e-2, help="Tolerance used by compare_shapes.")
    parser.add_argument("--out", type=str, default="/data-nas/data/experiments/genyue/code/genyue/neuralCAD-Edit/appendix/00000096_ours.png", help="Output image path.")
    parser.add_argument("--render-port", type=str, default=":99", help="X11 display port for offscreen render.")
    parser.add_argument("--azimuth", type=float, default=30.0, help="Camera azimuth in degrees.")
    parser.add_argument("--elevation", type=float, default=30.0, help="Camera elevation in degrees.")
    parser.add_argument("--resolution", type=int, default=512, help="Square output resolution.")
    parser.add_argument("--edge-width", type=float, default=2.0, help="Rendered edge width.")
    parser.add_argument(
        "--disable-lighting-shadows",
        action="store_true",
        help="Disable custom lighting and shadow/ray-tracing effects.",
    )
    parser.add_argument(
        "--default-view",
        action="store_true",
        help="Use renderer default view instead of azimuth/elevation camera.",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.gt_step and args.pred_step:
        result = render_pred_with_unmatched_faces(
            gt_step_path=args.gt_step,
            pred_step_path=args.pred_step,
            tol=args.tol,
            azimuth=args.azimuth,
            elevation=args.elevation,
            default_view=args.default_view,
            save_path=f"/tmp/{gen_uuid_4()}.png",
            render_port=args.render_port,
            resolution=args.resolution,
            edge_width=args.edge_width,
            material_style="plaster",
            disable_lighting_shadows=args.disable_lighting_shadows,
        )
        img = _base64_to_pil_image(result["image_base64"])
        img.save(out_path)
        print(f"Saved highlighted pred render to: {out_path}")
        print(
            "Compare result:",
            {
                "is_equal": result["is_equal"],
                "iou": result["iou"],
                "unmatched_pred_faces": result["unmatched_pred_faces"],
                "pred_face_count": result["pred_face_count"],
            },
        )
        print("Compare message:", result["compare_message"])
    else:
        step_path = args.step
        if not step_path:
            raise ValueError("Provide --step for plain render, or provide both --gt-step and --pred-step.")

        shape = _load_step_shape(step_path)
        img_b64, _ = render_brep(
            shape,
            azimuth=args.azimuth,
            elevation=args.elevation,
            default_view=args.default_view,
            brep_color="blue",
            save_path=f"/tmp/{gen_uuid_4()}.png",
            render_port=args.render_port,
            resolution=args.resolution,
            edge_width=args.edge_width,
            material_style="plaster",
            disable_lighting_shadows=args.disable_lighting_shadows,
        )
        img = _base64_to_pil_image(img_b64)
        img.save(out_path)
        print(f"Saved render to: {out_path}")