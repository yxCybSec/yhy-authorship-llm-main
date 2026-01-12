import os
import csv
import json
import time
import torch
import openai
import pickle
import random
import tiktoken
import py3langid
import numpy as np
import pandas as pd
import torch.nn.functional as F
import email
import warnings

# 设置HuggingFace镜像端点（用于加速模型下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("当前HF_ENDPOINT:", os.getenv("HF_ENDPOINT", "未设置"))

# 抑制transformers模型的初始化警告（这些警告不影响使用）
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


from random import shuffle
from sklearn import metrics
from ast import literal_eval
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

N_EVAL = 10  # 每次评估时从查询文本中采样的数量

#计算字符串的token数量
def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
#计算评估指标
def eval_fn(y_test, y_pred):
    acc = round(metrics.accuracy_score(y_test, y_pred)*100, 2)
    f1_w = round(metrics.f1_score(y_test, y_pred, average='weighted')*100, 2)
    f1_micro = round(metrics.f1_score(y_test, y_pred, average='micro')*100, 2)
    f1_macro = round(metrics.f1_score(y_test, y_pred, average='macro')*100, 2)
    return acc, f1_w, f1_micro, f1_macro
    
#将文本转换为嵌入向量
def embed_fn(model_name, texts, baseline_type):
    if baseline_type == 'bert':
        # 使用BERT系列模型生成嵌入
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        embedding = model(tokenized_texts.input_ids.to(model.device), tokenized_texts.attention_mask.to(model.device)).last_hidden_state.mean(dim=1)
    elif baseline_type == 'tf-idf':
        # 使用TF-IDF字符级n-gram特征
        vectorizer = TfidfVectorizer(max_features=3000, analyzer='char', ngram_range=(4, 4))
        embedding = torch.from_numpy(vectorizer.fit_transform(texts).toarray())
    elif baseline_type == 'ada':
        # 使用OpenAI的Ada嵌入模型
        ada_client = OpenAI(api_key="sk-wiwzcarugsitzxxbydhrgezmaifnztnjsfgzfgasgsimlobb",base_url="https://api.siliconflow.cn/v1")
        ada_response = ada_client.embeddings.create(input=texts, model="replace_this")
        embedding = torch.Tensor([e.embedding for e in ada_response.data])
    return embedding

#使用基线方法（嵌入相似度）进行作者归属
def run_aa_baseline(df_sub, model_name, baseline_type='bert'):
    ls_acc, ls_f1_w, ls_f1_micro, ls_f1_macro = [], [], [], []

    for i in df_sub.index:
        # 获取查询文本和候选作者文本
        ls_query_text, ls_potential_text = df_sub.loc[i, 'query_text'], df_sub.loc[i, 'potential_text']
        
        # 生成嵌入向量并归一化
        embed_query_texts = F.normalize(embed_fn(model_name, ls_query_text, baseline_type)) 
        embed_potential_texts = F.normalize(embed_fn(model_name, ls_potential_text, baseline_type))
        
        # 计算相似度矩阵（查询文本 × 候选作者文本）
        preds = embed_query_texts @ embed_potential_texts.T
        preds = F.softmax(preds, dim=-1)  # 归一化为概率分布
        
        # 真实标签：每个查询文本对应其索引位置（0, 1, 2, ...）
        labels = np.arange(0, len(ls_query_text))

        # 评估：预测作者ID = 最大相似度对应的索引
        acc, f1_w, f1_micro, f1_macro = eval_fn(labels, preds.argmax(-1).numpy())
        ls_acc.append(acc)
        ls_f1_w.append(f1_w)
        ls_f1_micro.append(f1_micro)
        ls_f1_macro.append(f1_macro)

    # 计算平均值和标准差
    muti_avg = (round(np.mean(ls_acc), 2), round(np.mean(ls_f1_w), 2), round(np.mean(ls_f1_micro), 2), round(np.mean(ls_f1_macro), 2))
    muti_std = (round(np.std(ls_acc), 2), round(np.std(ls_f1_w), 2), round(np.std(ls_f1_micro), 2), round(np.std(ls_f1_macro), 2))
    return muti_avg, muti_std

#使用LLM进行作者归属实验
def run_aa(df, method, model_name, prompt_input, system_msg, ls_df, ls_model, ls_method, n_eval=N_EVAL):
    start_time = time.time()
    df_res_all = pd.DataFrame()
    print("\n++++++++++ ", method, model_name, n_eval, " ++++++++++")

    for i in df.index:
        ls_reps = []
        text_label_map = {}  # 映射：查询文本 -> 作者ID（索引）
        sampled_queries = []  # 采样的查询文本列表
        
        # 获取当前样本的查询文本和候选作者文本
        ls_query_text, ls_potential_text = df.loc[i, 'query_text'], df.loc[i, 'potential_text']
        
        # 随机采样n_eval个查询文本进行评估
        random.seed(0)  # 固定随机种子以确保可复现
        for idx, val in random.sample(list(enumerate(ls_query_text)), n_eval):
            text_label_map[val] = idx  # 记录真实作者ID（索引位置）
            sampled_queries.append(val)
            
        # 对每个采样的查询文本进行推理
        for query_text in sampled_queries:
            # 将候选作者文本转换为字典格式（键为作者ID，值为文本）
            example_texts = json.dumps(dict(enumerate(ls_potential_text)))
            
            # 构建完整的prompt
            prompt = prompt_input+f"""The input texts are delimited with triple backticks. ```\n\nQuery text: {query_text} \n\nTexts from potential authors: {example_texts}\n\n```"""
            
            # 调用LLM API
            raw_response = client.chat.completions.create(
                model=model_name, 
                response_format={"type": "json_object"},  # 要求JSON格式输出
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ], 
                temperature=0  # 确定性输出
            )

            response_str = raw_response.choices[0].message.content
            print('\nRaw response content:\n', response_str, '\nLabel:', text_label_map[query_text])
            
            # 解析JSON响应
            try: 
                response = json.loads(response_str, strict=False)
            except json.JSONDecodeError:
                # JSON解析失败时的错误处理
                print(f"++++++++++ JSONDecodeError ++++++++++")
                response = json.loads("{}")
                response['analysis'] = response_str
                response['answer'] = -1  # 标记为无效答案

            # 记录结果
            response["query_text"] = query_text
            response["example_texts"] = example_texts
            response["tokens"] = raw_response.usage.total_tokens
            response["label"] = text_label_map[query_text]  # 真实作者ID
            ls_reps.append(response)
            response = None

        # 将当前样本的结果转换为DataFrame
        df_reps = pd.DataFrame(ls_reps)
        df_reps['answer'] = pd.to_numeric(df_reps['answer'], errors='coerce')  # 转换为数字
        df_reps['answer'] = df_reps['answer'].fillna(-1)  # 无效值填充为-1
        df_res_all = pd.concat([df_res_all, df_reps]).reset_index(drop=True)

    # 累积结果
    ls_df.append(df_res_all)
    ls_method.append(method)
    ls_model.append(model_name)
    print("--- Execution Time: %s seconds ---" % round(time.time() - start_time, 2))
    return df_res_all

#评估所有重复实验的结果，将结果按n_eval分组，每组代表一次重复实验，计算每组的评估指标，然后计算所有组的平均值和标准差
def eval_all_fn(df_res_all, n_eval):
    ls_acc, ls_f1_w, ls_f1_micro, ls_f1_macro = [], [], [], []
    
    # 按n_eval分组，每组代表一次重复实验
    for i in range(0, len(df_res_all.index), n_eval):
        df_reps = df_res_all[i: i+n_eval]
        acc, f1_w, f1_micro, f1_macro = eval_fn(df_reps["label"], df_reps["answer"])
        ls_acc.append(acc)
        ls_f1_w.append(f1_w)
        ls_f1_micro.append(f1_micro)
        ls_f1_macro.append(f1_macro)

    # 计算所有重复实验的平均值和标准差
    muti_avg = (round(np.mean(ls_acc), 2), round(np.mean(ls_f1_w), 2), round(np.mean(ls_f1_micro), 2), round(np.mean(ls_f1_macro), 2))
    muti_std = (round(np.std(ls_acc), 2), round(np.std(ls_f1_w), 2), round(np.std(ls_f1_micro), 2), round(np.std(ls_f1_macro), 2))
    return muti_avg, muti_std


dict_baseline = {
    'TF-IDF': 'TF-IDF', 
    'BERT': 'bert-base-uncased', 
    'RoBERTa': 'roberta-base', 
    'ELECTRA': 'google/electra-base-discriminator',
    'DeBERTa': 'microsoft/deberta-base'
}

# 嵌入类型字典：方法名称 -> 嵌入类型
dict_embed_type = {
    'TF-IDF': 'tf-idf', 
    'BERT': 'bert', 
    'RoBERTa': 'bert', 
    'ELECTRA': 'bert', 
    'DeBERTa': 'bert'
}
#比较基线方法和LLM方法的结果
def compare_baseline_mod(df_sub, ls_df, ls_model, ls_method, n_eval=N_EVAL, std_flag=False, baseline_idx=len(dict_baseline)):
    ls_res_avg, ls_res_std = [], []

    # 运行基线方法
    for key, val in list(dict_baseline.items())[:baseline_idx]:
        muti_avg, muti_std = run_aa_baseline(df_sub, val, dict_embed_type[key])
        ls_res_avg.append((key, val)+muti_avg+(0,))  # 基线方法没有"不确定"答案
        ls_res_std.append((key, val)+muti_std+(0,))

    # 评估LLM方法
    for i, df_tmp in enumerate(ls_df):
        muti_avg, muti_std = eval_all_fn(df_tmp, n_eval)
        answer_tmp = df_tmp.copy()
        
        # 统计"不确定"答案的数量（answer=-1）
        unsure_count = abs(answer_tmp[answer_tmp.answer==-1]['answer'].astype('int').sum())
        ls_res_avg.append((ls_method[i], ls_model[i])+muti_avg+(unsure_count,))
        ls_res_std.append((ls_method[i], ls_model[i])+muti_std+(None,))
    
    # 构建结果DataFrame
    res_avg = pd.DataFrame(ls_res_avg, columns=ls_col)
    res_std = pd.DataFrame(ls_res_std, columns=ls_col)
    
    if std_flag:
        return res_avg, res_std
    else:
        return res_avg

#从邮件对象中提取纯文本内容
def get_text_from_email(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)

#分离多个邮件地址
def split_email_addresses(line):
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

#博客数据集预处理
def prepare_blog_dataset(csv_path="D:\\authorship-llm-main\\data\\blogtext.csv", output_path=None):
    print("=" * 80)
    print("Data prep for the Blog dataset")
    print("=" * 80)
    
    # Read data
    df = pd.read_csv(csv_path)
    df.drop(['gender', 'age', 'topic', 'sign', 'date'], axis=1, inplace=True)
    print(f"Initial shape: {df.shape}")
    
    # Remove duplicates
    print('Before removing duplicates, df.shape:', df.shape)
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    print('After removing duplicates, df.shape:', df.shape)
    
    # Language filtering (English only)
    print(f"Filtering for English text...")
    df['lang'] = df['text'].apply(lambda x: py3langid.classify(x)[0])
    print(f"Total texts: {df.shape[0]:,}")
    print(f"English texts: {df[df.lang=='en'].shape[0]:,}")
    print(f"% of English text: {df[df.lang=='en'].shape[0] / df.shape[0]:.2%}")
    
    df = df[df.lang=='en']
    df.drop('lang', axis=1, inplace=True)
    print(f"After language filtering: {df.shape[0]:,}")
    
    # Filter by token count (< 512 tokens)
    print("Filtering texts with < 512 tokens...")
    df = df[df["text"].apply(lambda x: num_tokens_from_string(x, "gpt-3.5-turbo") < 512)]
    print(f"After token filter (<512): {df.shape[0]:,}")

    # Filter by token count (> 56 tokens)
    print("Filtering texts with > 56 tokens...")
    df = df[df["text"].apply(lambda x: num_tokens_from_string(x, "gpt-3.5-turbo") > 56)]
    print(f"After token filter (>56): {df.shape[0]:,}")

    # Keep only authors with at least 2 texts
    v = df.id.value_counts()
    df = df[df.id.isin(v[v >= 2].index)]
    print(f"# unique authors: {df.id.nunique()}")
    print(f"Final df.shape: {df.shape}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved processed dataset to {output_path}")

    return df

#邮件数据集预处理
def prepare_enron_email_dataset(csv_path="D:\\jupyter\\LLM\\authorship-llm-main\\data\\emails.csv", output_path=None):
    print("=" * 80)
    print("Data prep for the Enron Email dataset")
    print("=" * 80)

    # Read data
    emails_df = pd.read_csv(csv_path)
    print(f"Initial shape: {emails_df.shape}")

    # Parse emails
    print("Parsing email messages...")
    messages = list(map(email.message_from_string, emails_df['message']))
    for key in messages[0].keys():
        emails_df[key] = [doc[key] for doc in messages]

    emails_df['Text'] = list(map(get_text_from_email, messages))
    emails_df['From'] = emails_df['From'].map(split_email_addresses)
    emails_df['To'] = emails_df['To'].map(split_email_addresses)
    del messages

    emails_df = emails_df[['From', 'To', 'Text', 'Date', 'message']]

    # Handle multiple senders (take first one)
    emails_df['From'] = emails_df["From"].apply(lambda x: list(x)[0] if isinstance(x, frozenset) and len(x) > 0 else x)
    print(f"After parsing: {emails_df.shape}")

    # Remove duplicates
    print("Removing duplicates...")
    emails_df = emails_df.drop_duplicates(subset=['Text'], keep='first').reset_index(drop=True)
    print(f"After removing duplicates: {emails_df.shape}")

    # Create author ID mapping
    mail_corpus = emails_df.copy()
    mail_corpus.columns = ['user', 'receiver', 'text', 'date', 'message_old']
    unique_author = mail_corpus['user'].unique()
    email_mapping = {k: v for k, v in zip(unique_author, range(len(unique_author)))}
    mail_corpus['id'] = mail_corpus['user'].apply(lambda x: 'mail_'+str(email_mapping[x]))

    # Clean empty texts
    print(f"Empty texts: {mail_corpus[mail_corpus['text']==''].shape[0]}")
    mail_corpus.text = mail_corpus.text.apply(lambda x: x.strip() if isinstance(x, str) else '')

    # Select relevant columns
    df = mail_corpus[['text', 'id']].copy()
    print(f"After column selection: {df.shape}")

    # Remove duplicates again
    print('Before removing duplicates, df.shape:', df.shape)
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    print('After removing duplicates, df.shape:', df.shape)

    # Language filtering (English only)
    print(f"Filtering for English text...")
    df['lang'] = df['text'].apply(lambda x: py3langid.classify(x)[0] if x else 'unknown')
    print(f"Total texts: {df.shape[0]:,}")
    print(f"English texts: {df[df.lang=='en'].shape[0]:,}")
    print(f"% of English text: {df[df.lang=='en'].shape[0] / df.shape[0]:.2%}")

    df = df[df.lang=='en']
    df.drop('lang', axis=1, inplace=True)
    print(f"After language filtering: {df.shape[0]:,}")

    # Filter by token count (> 64 tokens)
    print("Filtering texts with > 64 tokens...")
    df = df[df["text"].apply(lambda x: num_tokens_from_string(x, "gpt-3.5-turbo") > 64)]
    print(f"After token filter (>64): {df.shape[0]:,}")
    
    # Keep only authors with at least 2 texts
    v = df.id.value_counts()
    df = df[df.id.isin(v[v >= 2].index)]
    print(f"# unique authors: {df.id.nunique()}")
    print(f"Final df.shape: {df.shape}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved processed dataset to {output_path}")
    
    return df

#为作者归属实验生成采样数据
def sampler_aa_fn_pro(df, n, reps):
    dict_to_df = []
    ls_unique_author = df.id.unique().tolist()
    
    for _ in range(reps):
        # 随机选择n个不同的作者
        candidate_authors = random.sample(ls_unique_author, n)
        # 从候选池中移除已选作者，确保每次重复实验的作者不重复
        ls_unique_author = [e for e in ls_unique_author if e not in candidate_authors]
        ls_queries, ls_potential_texts = [], []
        dict_row = {}
        
        for author_id in candidate_authors:
            # 获取该作者的所有文本
            author_texts = df.loc[author_id == df.id].text
            
            if len(author_texts) >= 2:
                # 如果作者有多篇文本，随机选择2篇
                text, text_same_author = author_texts.sample(2).values
                ls_queries.append(text)  # 查询文本
                ls_potential_texts.append(text_same_author)  # 候选文本
            else:
                # 如果作者只有一篇文本，则重复使用
                text = author_texts.iloc[0]
                ls_queries.append(text)
                ls_potential_texts.append(text)

        dict_row["query_text"] = ls_queries
        dict_row["potential_text"] = ls_potential_texts
        dict_to_df.append(dict_row)

    df_sub = pd.DataFrame(dict_to_df)
    return df_sub

if __name__ == "__main__":
    client = OpenAI(api_key="sk-wiwzcarugsitzxxbydhrgezmaifnztnjsfgzfgasgsimlobb",base_url="https://api.siliconflow.cn/v1")
    
    # 结果DataFrame的列名
    ls_col = ['Prompt', 'Model', 'Accuracy', 'Weighted F1', 'Micro F1', 'Macro F1', 'Unsure']

    # 模型配置
    m1 = "tencent/Hunyuan-MT-7B"  # 使用的LLM模型
    
    # 提示策略配置（4种不同的提示方法）
    v1, v2, v3, v4 = 'no_guidance', 'little_guidance', 'grammar', 'LIP'
    
    # 定义4种提示策略的prompt
    prompt1 = "Given a set of texts with known authors and a query text, determine the author of the query text. "  # 无指导
    prompt2 = prompt1+"Do not consider topic differences. "  # 少量指导
    prompt3 = prompt1+"Focus on grammatical styles. "  # 语法风格
    prompt4 = prompt1+"Analyze the writing styles of the input texts, disregarding the differences in topic and content. Focus on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. "  # LIP（语言信息提示）
    
    # 系统消息：定义LLM的输出格式（JSON格式，包含分析和答案）
    system_msg = """Respond with a JSON object including two key elements:
{
  "analysis": Reasoning behind your answer.
  "answer": The query text's author ID.
}"""


    # 预处理博客数据集
    blog_df = prepare_blog_dataset(csv_path="D:\\authorship-llm-main\\data\\blogtext.csv", output_path="../data/blog_processed.csv")
    blog_df_10 = sampler_aa_fn_pro(blog_df, n=10, reps=3)
    blog_df_10.to_csv("../data/blog_n10_reps3.csv", index=False)
    blog_df_20 = sampler_aa_fn_pro(blog_df, n=20, reps=3)
    blog_df_20.to_csv("../data/blog_n20_reps3.csv", index=False)

    # 预处理Enron邮件数据集
    email_df = prepare_enron_email_dataset(csv_path="D:\\authorship-llm-main\\data\\emails.csv", output_path="../data/email_processed.csv")
    email_df_10 = sampler_aa_fn_pro(email_df, n=10, reps=3)
    email_df_10.to_csv("../data/email_n10_reps3.csv", index=False)
    email_df_20 = sampler_aa_fn_pro(email_df, n=20, reps=3)
    email_df_20.to_csv("../data/email_n20_reps3.csv", index=False)


    # 运行实验
    print("=" * 80)
    print("实验：候选作者数量 n = 10")
    print("=" * 80)

    # 读取预处理好的数据（包含query_text和potential_text列，需要解析为列表）
    df_10 = pd.read_csv("../data/email_n10_reps3.csv", converters={"query_text": literal_eval, "potential_text": literal_eval})
    print(f"数据形状: {df_10.shape}, 候选作者数量: {len(df_10.loc[0, 'potential_text'])}")
    
    # 运行实验
    start_time = time.time()
    ls_df_10, ls_model_10, ls_method_10 = [], [], []  # 用于累积结果
    
    # 使用4种不同的提示策略运行实验
    df1_Hunyuan = run_aa(df_10, v1, m1, prompt1, system_msg, ls_df_10, ls_model_10, ls_method_10)  # 无指导
    df2_Hunyuan = run_aa(df_10, v2, m1, prompt2, system_msg, ls_df_10, ls_model_10, ls_method_10)  # 少量指导
    df3_Hunyuan = run_aa(df_10, v3, m1, prompt3, system_msg, ls_df_10, ls_model_10, ls_method_10)  # 语法风格
    df4_Hunyuan = run_aa(df_10, v4, m1, prompt4, system_msg, ls_df_10, ls_model_10, ls_method_10)  # LIP

    # 比较所有方法（包括基线方法和LLM方法）
    results_10 = compare_baseline_mod(df_10, ls_df_10, ls_model_10, ls_method_10)
    print(f"\n总执行时间: {time.time() - start_time:.2f} 秒")
    print("\nn=10 的结果:")
    print(results_10)
    








