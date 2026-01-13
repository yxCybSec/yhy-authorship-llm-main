import os
import csv
import json
import time
import random
import pickle
import openai
from openai import OpenAI
import tiktoken
import py3langid
import numpy as np
import pandas as pd
from sklearn import metrics

DATA_SIZE = 30

deploy_name_map = {
    "Hunyuan": "tencent/Hunyuan-MT-7B",
    "Qwen": "Qwen/Qwen3-8B"
}

official_name_map = {
    "Hunyuan": "tencent/Hunyuan-MT-7B",
    "Qwen": "Qwen/Qwen3-8B"
}

client = OpenAI(
    api_key="sk-wiwzcarugsitzxxbydhrgezmaifnztnjsfgzfgasgsimlobb",
    base_url="https://api.siliconflow.cn/v1"
)


# 评估函数
def eval_fn(y_test, y_pred, average='binary', print_flag=True):
    # 计算准确率
    acc = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
    # 计算F1分数
    f1 = round(metrics.f1_score(y_test, y_pred, average=average) * 100, 2)
    # 计算召回率
    recall = round(metrics.recall_score(y_test, y_pred, average=average) * 100, 2)
    # 计算精确率
    precision = round(metrics.precision_score(y_test, y_pred, average=average) * 100, 2)

    # 如果需要打印结果
    if print_flag:
        print(f"Accuracy: {acc} % | Precision: {precision} % | Recall: {recall} % | F1: {f1} %\n")

    return acc, precision, recall, f1


# 结果比较函数
def compare_baseline(exp_df_ls, exp_model_ls, exp_method_ls):
    ls_res = []
    # 遍历所有实验结果
    for i, df_tmp in enumerate(exp_df_ls):
        # 计算评估指标并拼接结果（方法名, 模型名, 评估指标, 数据形状）
        ls_res.append(
            (exp_method_ls[i], exp_model_ls[i]) +
            eval_fn(df_tmp["same"], df_tmp["answer"], print_flag=False) +
            df_tmp.shape  # 添加数据形状信息 (行数, 列数)
        )

    # 创建结果DataFrame并指定列名
    res = pd.DataFrame(
        ls_res,
        columns=['Prompt', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Size', 'df.shape[1]']
    )
    return res


#Token计数函数
def num_tokens_from_string(texts, model_id):
    # 获取对应模型的token编码方式
    encoding = tiktoken.encoding_for_model(model_id)
    # 编码文本并计算token数量
    num_tokens = len(encoding.encode(texts))
    return num_tokens


#核心验证函数
def run_verfication(df, method, model_name, prompt_prefix, system_msg, ls_df, ls_model, ls_method, prompt_postfix=""):
    # 存储单次实验结果的列表
    ls = []
    # 记录开始时间，用于计算执行耗时
    start_time = time.time()
    # 打印实验标识，方便控制台查看
    print(f"\n++++++++++ {method} {model_name} ++++++++++")

    # 遍历数据集中的每一行（每对文本）
    for i in df.index:
        # 获取作者ID和文本内容
        aut_id1, aut_id2 = df.loc[i, 'aut_id1'], df.loc[i, 'aut_id2']
        text1, text2 = df.loc[i, 'text1'], df.loc[i, 'text2']

        # 构建完整的用户提示词
        prompt = prompt_prefix + f"""The input texts (Text 1 and Text 2) are delimited with triple backticks. ```\n\nText 1: {text1}, \n\nText 2: {text2}\n\n```""" + prompt_postfix

        try:
            # 调用OpenAI API（实际是SiliconFlow平台）获取模型响应
            raw_response = client.chat.completions.create(
                model=deploy_name_map[model_name],  # 使用映射后的模型名
                # 指定JSON输出格式（仅对特定模型生效）
                response_format={"type": "json_object"} if model_name in ["Qwen", "Hunyuan"] else None,
                messages=[
                    {"role": "system", "content": system_msg},  # 系统角色提示
                    {"role": "user", "content": prompt}  # 用户输入提示
                ],
                temperature=0  # 温度设为0，保证输出确定性
            )

            # 提取模型响应内容
            response_str = raw_response.choices[0].message.content
            # 打印原始响应，方便调试
            print('Raw response content:', response_str, '\n')

            # 尝试解析JSON格式的响应
            try:
                response = json.loads(response_str, strict=False)
            except json.JSONDecodeError:
                # JSON解析失败时的容错处理
                print(f"===== JSONDecodeError =====\n")
                response = json.loads("{}")
                # 设置反向答案作为默认值（aut_id1==aut_id2的否定）
                response['answer'] = not (aut_id1 == aut_id2)
                # 记录错误信息和原始响应
                response['analysis'] = 'JSONDecodeError: ' + response_str

            # 补充原始数据到响应中，便于后续分析
            response["text1"], response["text2"] = text1, text2
            response["author_id1"], response["author_id2"] = aut_id1, aut_id2
            # 记录本次请求的token使用量
            response["tokens"] = raw_response.usage.total_tokens

            # 将本次结果添加到列表
            ls.append(response)
            # 清空response变量，避免内存泄漏
            response = None

        except Exception as e:
            # 捕获其他异常，保证程序不中断
            print(f"===== Exception at index {i}: {str(e)} =====\n")
            # 异常时添加空结果
            error_response = {
                'answer': False,
                'analysis': f'Error: {str(e)}',
                'text1': text1,
                'text2': text2,
                'author_id1': aut_id1,
                'author_id2': aut_id2,
                'tokens': 0
            }
            ls.append(error_response)

    # 将结果列表转换为DataFrame
    df_res = pd.DataFrame(ls)
    # 将结果添加到外部列表（用于后续汇总）
    ls_df.append(df_res)
    ls_method.append(method)
    ls_model.append(official_name_map[model_name])

    # 生成真实标签：判断两个作者ID是否相同
    df_res['same'] = df_res.author_id1 == df_res.author_id2
    # 确保预测答案为布尔类型（统一数据类型）
    df_res["answer"] = df_res["answer"].astype('bool')

    # 计算并打印本次实验的评估指标
    eval_fn(df_res["same"], df_res["answer"])
    # 打印本次实验的执行时间
    print(f"--- Execution Time: {round(time.time() - start_time, 2)} seconds ---")

    return df_res


#提示词配置
# 定义四种提示词方法的别名
v1, v2, v3, v4 = 'no_guidance', 'little_guidance', 'grammar', 'LIP'

system_msg = """
Respond with a JSON object including two key elements:
{
  "analysis": Reasoning behind your answer.
  "answer":  A boolean (True/False) answer.
}
"""

prompt1 = """
Verify if two input texts were written by the same author.
"""

prompt2 = """
Verify if two input texts were written by the same author. Analyze the writing styles of the input texts, disregarding the differences in topic and content.
"""

prompt3 = """
Verify if two input texts were written by the same author. Focus on grammatical styles indicative of authorship.
"""

prompt4 = """
Verify if two input texts were written by the same author. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. 
"""

def main():
    # 1. 读取实验数据集
    df_sub = pd.read_csv("../data/df_sub_blog_30.csv")
    # 打印数据集基本信息
    print("数据集形状:", df_sub.shape)
    print("数据集前5行:")
    print(df_sub.head())

    # 2. 第一次实验运行
    print("\n" + "=" * 50)
    print("第一次运行")
    print("=" * 50)
    start_time = time.time()
    # 初始化结果收集列表
    ls_df_1, ls_model_1, ls_method_1 = [], [], []

    # 运行四种提示词方法 × 两种模型的所有组合
    df1_Qwen = run_verfication(df_sub, v1, 'Qwen', prompt1, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df1_Hunyuan = run_verfication(df_sub, v1, 'Hunyuan', prompt1, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df2_Qwen = run_verfication(df_sub, v2, 'Qwen', prompt2, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df2_Hunyuan = run_verfication(df_sub, v2, 'Hunyuan', prompt2, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df3_Qwen = run_verfication(df_sub, v3, 'Qwen', prompt3, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df3_Hunyuan = run_verfication(df_sub, v3, 'Hunyuan', prompt3, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df4_Qwen = run_verfication(df_sub, v4, 'Qwen', prompt4, system_msg, ls_df_1, ls_model_1, ls_method_1)
    df4_Hunyuan = run_verfication(df_sub, v4, 'Hunyuan', prompt4, system_msg, ls_df_1, ls_model_1, ls_method_1)

    # 生成第一次运行的汇总结果
    res1 = compare_baseline(ls_df_1, ls_model_1, ls_method_1)
    print(f"第一次运行总执行时间: {time.time() - start_time:.2f} 秒")
    print("第一次运行结果:")
    print(res1)

    # 3. 第二次实验运行（重复实验以获取统计稳定性）
    print("\n" + "=" * 50)
    print("第二次运行")
    print("=" * 50)
    start_time = time.time()
    ls_df_2, ls_model_2, ls_method_2 = [], [], []

    # 重复运行所有实验组合
    df1_Qwen = run_verfication(df_sub, v1, 'Qwen', prompt1, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df1_Hunyuan = run_verfication(df_sub, v1, 'Hunyuan', prompt1, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df2_Qwen = run_verfication(df_sub, v2, 'Qwen', prompt2, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df2_Hunyuan = run_verfication(df_sub, v2, 'Hunyuan', prompt2, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df3_Qwen = run_verfication(df_sub, v3, 'Qwen', prompt3, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df3_Hunyuan = run_verfication(df_sub, v3, 'Hunyuan', prompt3, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df4_Qwen = run_verfication(df_sub, v4, 'Qwen', prompt4, system_msg, ls_df_2, ls_model_2, ls_method_2)
    df4_Hunyuan = run_verfication(df_sub, v4, 'Hunyuan', prompt4, system_msg, ls_df_2, ls_model_2, ls_method_2)

    res2 = compare_baseline(ls_df_2, ls_model_2, ls_method_2)
    print(f"第二次运行总执行时间: {time.time() - start_time:.2f} 秒")
    print("第二次运行结果:")
    print(res2)

    # 4. 第三次实验运行
    print("\n" + "=" * 50)
    print("第三次运行")
    print("=" * 50)
    start_time = time.time()
    ls_df_3, ls_model_3, ls_method_3 = [], [], []

    # 重复运行所有实验组合
    df1_Qwen = run_verfication(df_sub, v1, 'Qwen', prompt1, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df1_Hunyuan = run_verfication(df_sub, v1, 'Hunyuan', prompt1, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df2_Qwen = run_verfication(df_sub, v2, 'Qwen', prompt2, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df2_Hunyuan = run_verfication(df_sub, v2, 'Hunyuan', prompt2, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df3_Qwen = run_verfication(df_sub, v3, 'Qwen', prompt3, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df3_Hunyuan = run_verfication(df_sub, v3, 'Hunyuan', prompt3, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df4_Qwen = run_verfication(df_sub, v4, 'Qwen', prompt4, system_msg, ls_df_3, ls_model_3, ls_method_3)
    df4_Hunyuan = run_verfication(df_sub, v4, 'Hunyuan', prompt4, system_msg, ls_df_3, ls_model_3, ls_method_3)

    res3 = compare_baseline(ls_df_3, ls_model_3, ls_method_3)
    print(f"第三次运行总执行时间: {time.time() - start_time:.2f} 秒")
    print("第三次运行结果:")
    print(res3)

    # 5. 结果汇总与统计分析
    # 移除不需要的列（Size和df.shape[1]）
    r1 = res1.drop(['Size', 'df.shape[1]'], axis=1)
    r2 = res2.drop(['Size', 'df.shape[1]'], axis=1)
    r3 = res3.drop(['Size', 'df.shape[1]'], axis=1)

    # 合并三次运行的结果
    res_con = pd.concat([r1, r2, r3])

    # 按提示词和模型分组计算统计指标
    # 平均值（保留2位小数）
    res_mean = res_con.groupby(['Prompt', 'Model'], as_index=False, sort=False).mean().round(decimals=2)
    # 标准差（保留2位小数）
    res_std = res_con.groupby(['Prompt', 'Model'], as_index=False, sort=False).std().round(decimals=2)
    # 最大值（保留2位小数）
    res_max = res_con.groupby(['Prompt', 'Model'], as_index=False, sort=False).max().round(decimals=2)

    # 打印汇总结果
    print("\n" + "=" * 50)
    print("汇总结果 (平均值)")
    print("=" * 50)
    print(res_mean)

    print("\n" + "=" * 50)
    print("汇总结果 (平均值 ± 标准差)")
    print("=" * 50)
    # 拼接平均值和标准差，展示为 "均值±标准差" 格式
    print(res_mean.astype(str).iloc[:, 2:] + '±' + res_std.astype(str).iloc[:, 2:])


# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 调用主函数执行实验
    main()