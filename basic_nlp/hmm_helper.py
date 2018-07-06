import pickle
import codecs

PROB_START='start.py'
INPUT_DATA='../corpus/2014'+'/0101/c1002-23995935.txt'
PROB_EMIT='pro_emit.py'
PROB_TRANS='pro_trans.py'
class HMMHelper():
    def __init__(self):
        self.A_dic={}
        self.B_dic={}
        self.Pi_dic={}
        self.state_list=['B','M','E','S']
        self.state_num=4
        self.word_num=0
        self.A_row_count_dic = {}  # 记录A[key]所在行的所有列的总和，也就是在求解A_dic每行的每一个值候(A[key1][key2])，分母就是Count_dic[key]
        self.B_dic_element_size = {}  # 这个词典中记录的是，混淆矩阵中的对应着同一个输出字符的隐藏状态的数量
        # 初始化状态转移矩阵
        for state in self.state_list:
            self.A_dic[state]={}
            for state1 in self.state_list:
                self.A_dic[state][state1]=0.0
        # 初始化初始概率矩阵和混淆矩阵（发射矩阵）
        for state in self.state_list:
            self.Pi_dic[state]=0.0
            self.B_dic[state]={}

    def get_pos(self,input_str):
        output_str=[]
        input_str_length=len(input_str)
        if input_str_length==1:
            output_str.append('S')
        else:
            middle_num=input_str_length-2
            output_str.append('B')
            output_str.extend(['M']*middle_num)
            output_str.append('E')
        return output_str

    def main(self,train_file):
        train
        pass



