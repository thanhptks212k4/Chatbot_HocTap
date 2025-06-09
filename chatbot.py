import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import os
import re
import gradio as gr
import random

# PhoBERT

def initialize_phobert():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir="./phobert_cache")
    model = AutoModel.from_pretrained("vinai/phobert-base", cache_dir="./phobert_cache")
    return tokenizer, model

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return None
    return text.strip().replace('""', '"').replace('  ', ' ')

def get_phobert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    df["Câu hỏi"] = df["Câu hỏi"].apply(preprocess_text)
    df["Câu trả lời"] = df["Câu trả lời"].apply(preprocess_text)
    df = df.dropna(subset=["Câu hỏi"])
    return df

def build_faiss_index(df, tokenizer, model):
    embeddings = []
    valid_indices = []
    for idx, question in enumerate(df["Câu hỏi"]):
        emb = get_phobert_embedding(question, tokenizer, model)
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, valid_indices

def is_math_expression(text):
    return re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", text.strip()) is not None

def evaluate_math_expression(expr):
    try:
        result = eval(expr)
        return f"Câu trả lời : {result}"
    except:
        return "Không thể tính biểu thức."

def process_query(query, df, index, valid_indices, tokenizer, model, top_k=1):
    query = preprocess_text(query)
    if query is None:
        return "Câu hỏi không hợp lệ."

    query_emb = get_phobert_embedding(query, tokenizer, model)
    query_emb = np.array([query_emb]).astype('float32')
    _, indices = index.search(query_emb, top_k)

    responses = []
    for idx in indices[0]:
        original_idx = valid_indices[idx]
        answer = df.iloc[original_idx]["Câu trả lời"]
        solution = df.iloc[original_idx].get("Hướng dẫn giải", "")
        response = f"**Câu trả lời**: {answer}"
        if pd.notna(solution) and solution.strip():
            response += f"\n**Hướng dẫn giải**: {solution}"
        responses.append(response)

    return "\n\n".join(responses)

def get_random_exercise(df, selected_class, selected_subject):
    filtered = df[(df["Thể loại"] == "Bài tập") & (df["Lớp"] == selected_class) & (df["Môn học"] == selected_subject)]
    if len(filtered) == 0:
        return None
    return filtered.sample(1).iloc[0]

def chatbot_theory_interface(query):
    if is_math_expression(query):
        return evaluate_math_expression(query)
    return process_query(query, df_global, index_global, valid_indices_global, tokenizer_global, model_global)

def evaluate_answer(user_answer, correct_answer, solution=""):
    if user_answer.strip().lower() == correct_answer.strip().lower():
        return f"✅ Chính xác!\n\n**Đáp án**: {correct_answer}\n\n**Hướng dẫn giải**: {solution}"
    else:
        return f"❌ Chưa đúng.\n\n**Đáp án đúng**: {correct_answer}\n\n**Hướng dẫn giải**: {solution}"

def launch_gradio_chatbot(csv_path="E:\chatbot\chatbot.csv"):
    global tokenizer_global, model_global, df_global, index_global, valid_indices_global
    tokenizer_global, model_global = initialize_phobert()
    df_global = load_and_preprocess_data(csv_path)
    index_global, valid_indices_global = build_faiss_index(df_global, tokenizer_global, model_global)

    classes = sorted(df_global["Lớp"].dropna().unique().tolist())
    subjects = sorted(df_global["Môn học"].dropna().unique().tolist())

    with open("chatbott.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    with open("chatbott.css", "r", encoding="utf-8") as f:
        css_content = f.read()

    with gr.Blocks(css=css_content) as demo:
        gr.HTML(html_content)
        category = gr.Radio(["Yêu cầu câu hỏi", "Yêu cầu bài tập"], label="Chọn thể loại", value="Lý thuyết")
        class_dropdown = gr.Dropdown(choices=classes, label="Chọn lớp", visible=False)
        subject_dropdown = gr.Dropdown(choices=subjects, label="Chọn môn học", visible=False)

        theory_input = gr.Textbox(label="Nhập câu hỏi", lines=2, visible=True)
        theory_output = gr.Markdown()

        exercise_display = gr.Textbox(label="Câu hỏi bài tập", lines=3, interactive=False, visible=False)
        user_answer = gr.Textbox(label="Nhập câu trả lời của bạn", visible=False)
        check_button = gr.Button("Kiểm tra", visible=False)
        exercise_output = gr.Markdown()
        current_question = gr.State()

        def toggle_category_inputs(selected):
            return (
                gr.update(visible=(selected == "Yêu cầu bài tập")),
                gr.update(visible=(selected == "Yêu cầu bài tập")),
                gr.update(visible=(selected == "Yêu cầu câu hỏi")),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )

        category.change(fn=toggle_category_inputs, inputs=[category],
                        outputs=[class_dropdown, subject_dropdown, theory_input, exercise_display, user_answer, check_button])

        theory_input.submit(fn=chatbot_theory_interface, inputs=theory_input, outputs=theory_output)

        def load_exercise(selected_class, selected_subject):
            ex = get_random_exercise(df_global, selected_class, selected_subject)
            if ex is None:
                return "Không tìm thấy bài tập phù hợp.", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return ex["Câu hỏi"], ex, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        class_dropdown.change(fn=load_exercise, inputs=[class_dropdown, subject_dropdown],
                              outputs=[exercise_display, current_question, exercise_display, user_answer, check_button])
        subject_dropdown.change(fn=load_exercise, inputs=[class_dropdown, subject_dropdown],
                              outputs=[exercise_display, current_question, exercise_display, user_answer, check_button])

        def check_exercise_answer(user_ans, ex):
            if ex is None:
                return "Không có bài tập nào đang hoạt động."
            return evaluate_answer(user_ans, ex["Câu trả lời"], ex.get("Hướng dẫn giải", ""))

        check_button.click(fn=check_exercise_answer, inputs=[user_answer, current_question], outputs=exercise_output)

    demo.launch()

if __name__ == "__main__":
    launch_gradio_chatbot()