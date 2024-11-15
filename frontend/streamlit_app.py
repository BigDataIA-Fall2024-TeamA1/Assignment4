# streamlit.py

import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目的根目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以从 backend 目录中导入 agents 模块
from backend.agents import ask_question

import streamlit as st

st.title("学术研究助手")

# 会话状态存储对话历史
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 文档选择（假设您有一个文档列表）
# 您可以根据实际情况从数据库或文件中读取文档列表
documents = ["Document1", "Document2", "Document3"]
selected_document = st.selectbox("请选择要研究的文档：", documents)

st.write(f"您选择的文档是：{selected_document}")

# 问题输入
question = st.text_input("请针对选定的文档提问：")

if st.button("提交"):
    if question:
        with st.spinner("思考中..."):
            # 将文档上下文添加到问题中
            question_with_context = f"Document: {selected_document}\nQuestion: {question}"
            answer = ask_question(question_with_context)
            st.session_state.conversation.append((question, answer))
            st.success("回答已生成！")
    else:
        st.warning("请输入您的问题。")

# 显示对话历史
st.subheader("对话历史")
for i, (q, a) in enumerate(st.session_state.conversation):
    st.write(f"**问题 {i+1}：** {q}")
    st.write(f"**回答 {i+1}：** {a}")

# 保存研究会话结果
if st.button("保存会话"):
    import json
    session_data = {
        "document": selected_document,
        "conversation": st.session_state.conversation
    }
    with open("session_results.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)
    st.success("会话已成功保存！")

# 导出为专业的 PDF 报告
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

if st.button("导出为 PDF"):
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"关于 {selected_document} 的研究报告", styles['Title']))
    elements.append(Spacer(1, 12))

    for i, (q, a) in enumerate(st.session_state.conversation):
        elements.append(Paragraph(f"问题 {i+1}：{q}", styles['Heading2']))
        elements.append(Paragraph(f"回答 {i+1}：{a}", styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)

    buffer.seek(0)
    st.download_button(
        label="下载 PDF 报告",
        data=buffer,
        file_name="research_report.pdf",
        mime="application/pdf"
    )
