import random
import torch
import torch.nn.functional as F

from .model import conversation as conversation_lib
from .utils_llcb import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, LONG_QUESTION_LIST,)

def transfer(text_list):
    long_question_list = LONG_QUESTION_LIST
    answer_list = ANSWER_LIST
    questions = []
    answers = []
    #当textlist只有一句话时，他的长度就变成了这句话字符的长度，而不是list的长度了，那么后面也就出错了
    for text in text_list:
        question_template = random.choice(long_question_list)
        questions.append(question_template.format(sent=text))
        answers.append(random.choice(answer_list).format(sent=text))
    conversations = []
    conv = conversation_lib.conv_llava_v1.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    i = 0
    # import pdb; pdb.set_trace()
    while i < len(questions):
        conv.messages = []
        conv.append_message(conv.roles[0], questions[i])
        conv.append_message(conv.roles[1], answers[i])
        conversations.append(conv.get_prompt())
        i += 1
    # import pdb; pdb.set_trace()
    # print("conversation的内容为", conversations)
    # print("conversation的类型为", conversations[0].__class__)
    
    return (
        conversations,
        questions
    )

if __name__ == '__main__':
# Example usage
    text_list = ["stack the blocks"]
    # text_list = ["stack the blocks", "sort the objects by color", "arrange the books alphabetically"]
    conversations, questions = transfer(text_list)
    print(conversations)
    print(questions)
