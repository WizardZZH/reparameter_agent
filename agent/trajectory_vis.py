import re
import os
from datetime import datetime
import json

tool_template = """
Use the tool:
    {tool_name}
Input:
    {inputs}
""".strip()

def parse_message(msg):
    """解析消息字符串，提取关键信息"""
    # 使用正则表达式提取字段
    dialogue = []
    
    role = msg.role
    assert role in ['assistant', 'user', "system", "tool"], f"Unknown role: {role}"
    name = msg.name
    content = msg.content
    time_stamp = msg.timestamp


    if isinstance(content, str):
            item_type = 'text'
            content = content
            dialogue.append(
                {
                    'role': role,
                    'name': name,
                    'type': item_type,
                    'content': content,
                    'timestamp': time_stamp
                }
            )

    else:
        for item in content:
            if item["type"] == "text":
                item_type = 'text'
                content = item['text']
                dialogue.append(
                    {
                        'role': role,
                        'name': name,
                        'type': item_type,
                        'content': content,
                        'timestamp': time_stamp
                    }
                )

            elif item["type"] == "thinking":
                item_type = 'thinking'
                content = item.get('thinking', '')
                dialogue.append(
                    {
                        'role': role,
                        'name': name,
                        'type': item_type,
                        'content': content,
                        'timestamp': time_stamp
                    }
                )

            elif item["type"] == "tool_use":
                item_type = 'text'
                if isinstance(item["input"],dict):
                    for k,v in item["input"].items():
                        if isinstance(v,str):
                            item["input"][k] = v.replace("\n","\\n")

                    content = tool_template.format(tool_name = item["name"], inputs = json.dumps(item["input"], ensure_ascii=False, indent=4).replace("\\n","\n"))
                else:
                    content = tool_template.format(tool_name = item["name"], inputs = str(item["input"]))

                dialogue.append(
                    {
                        'role': role,
                        'name': name,
                        'type': item_type,
                        'content': content,
                        'timestamp': time_stamp
                    }
                )
            elif item["type"] == "image":
                item_type = 'image'
                source = item["source"]
                content = f"data:{source['media_type']};{source['type']},{source['data']}"
                dialogue.append(
                    {
                        'role': role,
                        'name': name,
                        'type': item_type,
                        'content': content,
                        'timestamp': time_stamp
                    }
                )


            elif item["type"] == "tool_result":
                for output_item in item["output"]:
                    if output_item["type"] == "text":
                        item_type = 'text'
                        content = output_item['text']
                        dialogue.append(
                            {
                                'role': role,
                                'name': name,
                                'type': item_type,
                                'content': content,
                                'timestamp': time_stamp
                            }
                        )
                    if output_item["type"] == "image":
                        item_type = 'image'
                        source = output_item["source"]
                        content = f"data:{source['media_type']};{source['type']},{source['data']}"
                        dialogue.append(
                            {
                                'role': role,
                                'name': name,
                                'type': item_type,
                                'content': content,
                                'timestamp': time_stamp
                            }
                        )
    return dialogue

def generate_html(messages, title):
    """生成HTML可视化页面"""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>对话可视化</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .chat-container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 15px;
            max-width: 70%;
            word-wrap: break-word;
        }}
        .assistant {{
            background-color: white;
            border: 1px solid #ddd;
            margin-right: auto;
            text-align: left;
        }}
        .system {{
            background-color: white;
            border: 1px solid #ddd;
            margin-right: auto;
            text-align: left;
        }}
        .user {{
            background-color: #e0f7ea;
            border: 1px solid #b3e6cc;
            margin-left: auto;
            text-align: left;
        }}
        .name {{
            font-weight: bold;
            color: black;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        .content {{
            color: black;
            line-height: 1.4;
            white-space: pre-wrap; /* 保留空白符，但允许换行 */
            word-wrap: break-word; /* 允许长单词换行 */
            overflow-wrap: break-word; /* 更现代的换行属性 */
        }}
        .content-thinking {{
            color: #5a4b00;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
            background: #f2f2f2;
            border: 1px dashed #d5b95c;
            border-radius: 8px;
            padding: 10px;
        }}
        .timestamp {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }}
        .message-image {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>💬 {title} </h1>
    </div>
    <div class="chat-container">
        {messages_html}
    </div>
</body>
</html>
    """
    
    messages_html = ""
    for msg in messages:
        # 确定消息样式类
        message_class = msg["role"]

        if msg["type"] == "text":
            messages_html += f"""
            <div class="message {message_class}">
                <div class="name">{msg['name']}</div>
                <div class="content">{msg['content']}</div>
                <div class="timestamp">{msg['timestamp']}</div>
            </div>
            """
        elif msg["type"] == "thinking":
            messages_html += f"""
            <div class="message {message_class}">
                <div class="name">{msg['name']} (thinking)</div>
                <div class="content-thinking">{msg['content']}</div>
                <div class="timestamp">{msg['timestamp']}</div>
            </div>
            """
        elif msg["type"] == "image":
            messages_html += f"""
            <div class="message {message_class}">
                <div class="name">{msg['name']}</div>
                <div class='content'><img class='message-image' src='{msg['content']}' alt='图像内容'></div>
                <div class="timestamp">{msg['timestamp']}</div>
            </div>
            """
    
    return html_template.format(title = title, messages_html=messages_html)

def process_conversation(msgs, title, out_path):
    """处理对话文件并生成HTML"""
    messages = []
    for msg in msgs:
        dialogue = parse_message(msg)
        messages.extend(dialogue)
    # 生成HTML
    html_content = generate_html(messages, title=title)
    
    # 保存HTML文件
    output_file = os.path.join(out_path, f"{title}.html")

    name = f"{title}.html"

    root = out_path.split("/")[3:]
    root = "/".join(root)
    print(f"Saving HTML to:  {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
