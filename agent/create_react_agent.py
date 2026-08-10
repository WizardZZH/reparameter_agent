import asyncio
from json import tool
import yaml
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.mcp import HttpStatelessClient
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from agentscope.plan import PlanNotebook
from agentscope.token import CharTokenCounter
from agentscope.tool import (
    view_text_file,
    execute_python_code,
    execute_shell_command,
    write_text_file,
    insert_text_file
)

from typing import  Any
from agentscope.plan import PlanNotebook, SubTask



async def load_agent_from_config(config: Any) -> ReActAgent:
    # load config from yaml file
    
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Config should be a dict or a path to a yaml file.")

    # support wrapped and flat config styles
    root_cfg = config.get("agent_config", config)

    agent_args = root_cfg.get("agent_args", {})
    model_args = root_cfg.get("model_args", {})
    raw_tool_args = root_cfg.get("tool_args", None)
    tool_args = []
    client_args = root_cfg.get("client_args", {})
    client_id = root_cfg.get("client_id", None)
    runtime_args = root_cfg.get("runtime_args", {})

    # optional formatter mapping
    formatter_name = runtime_args.get("formatter", "OpenAIChatFormatter")
    formatter_map = {
        "OpenAIChatFormatter": OpenAIChatFormatter(),
    }
    formatter = formatter_map.get(formatter_name, OpenAIChatFormatter())
    formatter.promote_tool_result_images = True
    
    # Support both a single toolkit dict and a list of toolkit dicts.
    if isinstance(raw_tool_args, dict):
        raw_tool_args = [raw_tool_args]

    if isinstance(raw_tool_args, list):
        for cfg in raw_tool_args:
            if isinstance(cfg, dict) and cfg.get("toolkit_name") and cfg.get("toolkit_url"):
                tool_args.append(cfg)

    # if tool_args is empty/incomplete, skip MCP registration
    if not tool_args:
        tool_args = None
    
    agent = await create_agent(
        agent_args=agent_args,
        model_args=model_args,
        tool_args=tool_args,
        client_args=client_args,
        client_id=client_id,
        formatter=formatter,
        skills=runtime_args.get("skills", None),
        sys_tools=runtime_args.get("sys_tools", None),
        function_tools=runtime_args.get("function_tools", None),
        print_hint_msg=runtime_args.get("print_hint_msg", True),
    )

    return agent
        
        
    
    
from pydantic import BaseModel, Field

# 定义自定义摘要结构
class CustomSummary(BaseModel):
    main_topic: str = Field(
        max_length=2000,
        description="对话的主题"
    )
    key_points: str = Field(
        max_length=4000,
        description="讨论的重要观点"
    )
    pending_tasks: str = Field(
        max_length=2000,
        description="待完成的任务"
    )



async def create_agent(
    agent_args:dict,
    model_args:dict,
    tool_args:Any = None,
    client_args:dict = None,
    client_id: str = None,
    formatter = OpenAIChatFormatter(),
    memory = None,
    skills: list[str] = None,
    sys_tools: list[str] = None,
    function_tools: list[Any] = None,
    print_hint_msg: bool = True,
    ):
    
    
    if client_args is None:
        client_args = {}

    # 1. create model
    model_name = model_args.get("model_name", "gemini-3-flash-preview")
    api_key = model_args.get("api_key", "sk-JJqe7SC9LUO0QVYxFe2eBd71F3E646Fc9b34B972Fe0436A7")
    stream = model_args.get("stream", True)
    
    default_generate_kwargs={
            "temperature": 1.0,
            "max_tokens": 6000,
            "seed": 42,   
            "parallel_tool_calls": False,
        }
    generate_kwargs = model_args.get("generate_kwargs", default_generate_kwargs)
    
    if api_key == "EMPTY":
        host_address = model_args.get("host", "localhost")
        server_port = model_args.get("server_port", 8000)
        model = OpenAIChatModel(
            model_name=model_name,
            api_key="EMPTY",
            client_kwargs={"base_url": f"http://{host_address}:{server_port}/v1"},
            generate_kwargs=generate_kwargs
            )
    else:
        model = OpenAIChatModel(
                model_name=model_name,
                reasoning_effort = model_args.get("reasoning_effort", None),
                api_key=api_key,
                stream=stream,
                client_kwargs=client_args,
                generate_kwargs=generate_kwargs
            )    
    
    # 2. prepare context
    in_memory = InMemoryMemory()       
    if memory is not None:
        await in_memory.add(memory)

    plan_notebook = None

        
    # 3. prepare toolkit
    toolkit = Toolkit()
    
    # 3.1 prepara mcp tools
    if tool_args is not None:
        tool_cfgs = tool_args if isinstance(tool_args, list) else [tool_args]

        for tool_cfg in tool_cfgs:
            assert "toolkit_name" in tool_cfg, "tool_args should include 'toolkit_name'"
            assert "toolkit_url" in tool_cfg, "tool_args should include 'toolkit_url'"

            toolkit_name = tool_cfg.get("toolkit_name")
            toolkit_url = tool_cfg.get("toolkit_url")
            activate_tools = tool_cfg.get("activate_tools", None)

            if client_id is not None:
                toolkit_client = HttpStatelessClient(
                    name=toolkit_name,
                    transport="sse",
                    url=toolkit_url, 
                    headers={"header-client-id": client_id},# 确保MCP服务器正在运行
                )
            else:
                toolkit_client = HttpStatelessClient(
                    name=toolkit_name,
                    transport="sse",
                    url=toolkit_url, # 确保MCP服务器正在运行
                )
            
            
            if activate_tools is not None:
                await toolkit.register_mcp_client(
                    toolkit_client,
                    enable_funcs= activate_tools,
                )
            else:
                await toolkit.register_mcp_client(
                    toolkit_client,
                )
            
    # 3.2 register system tools and skills
    if skills is not None:
        if "view" in sys_tools:
            toolkit.register_tool_function(view_text_file)

        if "write" in sys_tools:
            toolkit.register_tool_function(write_text_file)
            toolkit.register_tool_function(insert_text_file)
        
        if "execute" in sys_tools:
            toolkit.register_tool_function(execute_python_code)
            toolkit.register_tool_function(execute_shell_command)
        
        skill_tool_path = ".agents/skills/"
        for skill in skills:
            toolkit.register_agent_skill(skill_tool_path + skill)
    
    # 3.3 register function tools
    if function_tools is not None: 
        import importlib
        for func_tool_name, func_tool_args in function_tools:
            module = importlib.import_module(func_tool_args.get("module"))
            function_tool = getattr(module, func_tool_args.get("func_spec"))
            toolkit.register_tool_function(function_tool)
    
    # 4. initialize agent
    assert "sys_prompt" in agent_args, "agent_args should include 'sys_prompt'"
    assert "agent_name" in agent_args, "agent_args should include 'agent_name'"
    
    
    agent_name = agent_args.get("agent_name")
    sys_prompt = agent_args.get("sys_prompt")
    max_iters = agent_args.get("max_iters", 5)
    
    compress_config = agent_args.get("compress_config", None)
    if compress_config is None:
        compression_config = None
    else:   
        # compression_config=ReActAgent.CompressionConfig(
        #         enable=True,
        #         agent_token_counter=CharTokenCounter(),
        #         trigger_threshold=compress_config.get("trigger_threshold", 10000),
        #         keep_recent=compress_config.get("keep_recent", 10),
        #     )
        
        compression_config=ReActAgent.CompressionConfig(
            enable=True,
            agent_token_counter=CharTokenCounter(),
            trigger_threshold=compress_config.get("trigger_threshold", 10000),
            keep_recent=compress_config.get("keep_recent", 5),
            # 结构化摘要的自定义 schema
            summary_schema=CustomSummary,
            # 指导压缩的自定义提示
            compression_prompt=(
                "<system-hint>请总结上述对话，"
                "重点关注主题、关键讨论点和待完成任务。</system-hint>"
            ),
            # 格式化摘要的自定义模板
            summary_template=(
                "<system-info>对话摘要：\n"
                "主题：{main_topic}\n\n"
                "关键观点：\n{key_points}\n\n"
                "待完成任务：\n{pending_tasks}"
                "</system-info>"
            ),
        )
    

    agent = ReActAgent(
            name= agent_name,
            sys_prompt=sys_prompt,
            model=model,
            memory=in_memory,
            formatter=formatter,
            toolkit=toolkit,
            plan_notebook=plan_notebook,
            print_hint_msg=print_hint_msg,
            max_iters=max_iters,
            compression_config=compression_config,
        )
    
    
    return agent


