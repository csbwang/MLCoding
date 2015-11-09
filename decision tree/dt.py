# -*- coding:utf-8 -*-
__author__ = 'phenix'


import math
import operator


class DecisionTree(object):
	"""
	决策树分类算法
	"""
	def __init__(self, features=[], discretization=False, alpha_type=0):
		"""
		初始默认为离散属性
		:param features:属性标签列表，默认为空，训练时生成
		:param discretization: 是否需要离散化属性, 默认不需要
		:param alpha_type: 离散属性时选取的阈值类型，0为平均数，1为中位数，2为众数，默认为 0
		:return:
		"""
		self.features = features
		self.discretization = discretization
		self.alpha_type = alpha_type

	def __calculate_shannon_entropy(self, data_set):
		"""
		计算香农熵
		:param data_set:
		:return:shannon_ent
		"""
		num_entries = len(data_set)  # 实例的个数
		class_counts = {}
		for feature_vec in data_set:  # 遍历每个实例，统计标签的频次
			current_class = feature_vec[-1]
			if current_class not in class_counts.keys():
				class_counts[current_class] = 0
			class_counts[current_class] += 1
		shannon_entropy = 0.0
		for key in class_counts:
			prob = float(class_counts[key])/num_entries
			shannon_entropy -= prob * math.log(prob, 2)
		return shannon_entropy

	def __split_data_set(self, data_set, axis, feature_value):
		"""
		使用feature特征划分数据集,这里的划分实际上是从data_set中去除feature特征
		:param data_set:
		:param axis:特征维度
		:param feature_value:某一特征维度的一个特征值
		:return:符合该特征的所有实例（并且自动移除掉这维特征）
		"""
		new_data_set = []
		for feature_vec in data_set:
			if feature_vec[axis] == feature_value:
				new_feature_vec = feature_vec[: axis]
				new_feature_vec.extend(feature_vec[axis + 1:])
				new_data_set.append(new_feature_vec)
		return new_data_set

	def __calculate_conditional_entropy(self, data_set, i, unique_feature_values):
		"""
		计算x_i给定的条件下，y的条件熵
		:param data_set:
		:param i: 特征维度i
		:param unique_feature_values:特征维度i的所有特征值
		:return conditional_entropy：特征维度i的条件熵
		"""
		conditional_entropy = 0.0
		for feature_value in unique_feature_values:  # 遍历当前特征中的所有唯一特征值
			sub_data_set = self.__split_data_set(data_set, i, feature_value)
			prob = len(sub_data_set) / (len(data_set) * 1.0)  # 极大似然估计概率
			conditional_entropy += prob * self.__calculate_shannon_entropy(sub_data_set)  # ∑pH(Y|X=xi) 条件熵的计算
		return conditional_entropy

	def __calculate_information_gain(self, data_set, base_entropy, i):
		"""
		计算信息增益
		:param data_set:
		:param base_entropy: 数据集中Y的信息熵
		:param i: 特征维度i
		:return: 特征i对数据集的信息增益g(data_set|x_i)
		"""
		feature_values = [example[i] for example in data_set]
		unique_feature_values = set(feature_values)
		current_conditional_entropy = self.__calculate_conditional_entropy(data_set, i, unique_feature_values)
		information_gain = base_entropy - current_conditional_entropy  # 信息增益，就是熵的减少，也就是不确定性的减少
		return information_gain

	def __chose_best_feature_to_split(self, data_set):
		num_feature = len(data_set[0]) - 1  # 最后一列是分类
		base_entropy = self.__calculate_shannon_entropy(data_set)
		best_information_gain = -1
		# best_feature_dimension = 0 #--------------------------------------------
		for i in range(num_feature):  # 遍历所有维度特征
			information_gain = self.__calculate_information_gain(data_set, base_entropy, i)
			if information_gain > best_information_gain:
				best_information_gain = information_gain
				best_feature_dimension = i
		return best_feature_dimension  # 返回最佳特征对应的维度

	def __majority_vote(self, class_list):
		"""
		返回出现次数最多的类别名称
		:param class_list:
		:return:
		"""
		class_count = {}
		for vote in class_list:
			if vote not in class_count.keys():
				class_count[vote] = 0
			class_count[vote] += 1
		sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)  # 返回键值对的列表
		return sorted_class_count[0][0]

	def __create_decision_tree(self, data_set, features):
		"""
		创建决策树
		:type data_set: 二维列表
		:param data_set:
		:param features:所有特征的标签列表，作为树中每个非叶子节点的标记
		"""
		class_list = [example[-1] for example in data_set]
		if class_list.count(class_list[0]) == len(class_list):
			return class_list[0]  # 当类别完全相同则停止继续划分，返回该类别标签
		if len(data_set[0]) == 1:
			return self.__majority_vote(class_list)  # 当使用完所有特征后停止继续划分，使用多数表决返回类别标签
		best_feature_dimension = self.__chose_best_feature_to_split(data_set)
		if best_feature_dimension == -1:
			return self.__majority_vote(class_list)
		best_feature = features[best_feature_dimension]
		decision_tree = {best_feature: {}}
		del(features[best_feature_dimension])  # 删除该特征
		feature_values = [example[best_feature_dimension] for example in data_set]
		unique_feature_values = set(feature_values)
		for value in unique_feature_values:
			sub_features = features[:]  # 复制列表
			decision_tree[best_feature][value] = self.__create_decision_tree(
				self.__split_data_set(data_set, best_feature_dimension, value), sub_features)
		return decision_tree

	def __classify(self, decision_tree, test_vec):
		"""
		使用生成的决策树对test_vec分类
		:param decision_tree:
		:param test_vec:
		:return:test_vec的类别标签
		"""
		first_str = decision_tree.keys()[0]
		second_dict = decision_tree[first_str]
		feature_index = self.features.index(first_str)
		for key in second_dict.keys():
			if test_vec[feature_index] == key:
				if type(second_dict[key]).__name__ == 'dict':
					class_label = self.__classify(second_dict[key], test_vec)
				else:
					class_label = second_dict[key]
		return class_label

	def __feature_discretization(self, data_set):
		"""
		连续属性离散化
		:param data_set:
		:return:
		"""
		for i in range(len(data_set)):
			for j in range(len(self.alpha_list)):
				if float(data_set[i][j]) < self.alpha_list[j][self.alpha_type]:
					data_set[i][j] = 0
				else:
					data_set[i][j] = 1

	def __get_discretization_threshold(self, data_set):
		"""
		生成每个特征属性离散化的阈值
		:param data_set:
		:return:
		"""
		entries_nums = len(data_set)
		dimension = len(data_set[0]) - 1
		alpha_list = []  # 离散化时每个特征的的阈值
		for i in range(dimension):
			feature_i_values = [float(data_set[j][i]) for j in range(entries_nums)]
			feature_i_values.sort()
			average = sum(feature_i_values) / (entries_nums * 1.0)  # 平均数
			median = feature_i_values[entries_nums / 2]  # 中位数
			mode_num = 0
			mode = feature_i_values[0]
			unique_feature_i_values = set(feature_i_values)
			for item in unique_feature_i_values:
				if feature_i_values.count(item) > mode_num:
					mode = item  # 众数
			alpha_list.append([average, median, mode])
		self.alpha_list = alpha_list

	def fit(self, train_data_set):
		"""
		训练决策树
		:param train_data_set:
		:param discretization:是否需要属性离散化
		:return:
		"""
		if self.discretization:
			self.__get_discretization_threshold(train_data_set)  # 生成每个特征属性离散化的阈值
			self.__feature_discretization(train_data_set)
		if not self.features:
			self.features = range(1, len(train_data_set[0]))
		features = self.features[:]
		self.decision_tree = self.__create_decision_tree(train_data_set, features)

	def predict(self, test_data_set):
		"""
		使用生成的决策树对test_data_set分类
		:param test_data_set:
		:return:
		"""
		if self.discretization:
			self.__feature_discretization(test_data_set)
		class_label_list = []
		for i in range(len(test_data_set)):
			class_label_list.append(self.__classify(self.decision_tree, test_data_set[i]))
		return class_label_list


def load_data(train_file_path, test_file_path):
	"""
	加载数据，生成训练数据和测试数据
	:param train_file_path:
	:param test_file_path:
	:return:
	"""
	train_data_set = []
	with open(train_file_path) as train_file:
		for line in train_file:
			split_list = line.strip().split(',')
			train_data_set.append(split_list)
	class_label_list = []
	test_data_set = []
	with open(test_file_path) as test_file:
		for line in test_file:
			split_list = line.strip().split(',')
			class_label_list.append(split_list[-1])
			test_data_set.append(split_list[:-1])
	return train_data_set, test_data_set, class_label_list


def main(train_file_path, test_file_path):
	features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
	clf = DecisionTree(features, True, 1)  # 初始化决策树模型实例
	train_data_set, test_data_set, true_class_label_list = load_data(train_file_path, test_file_path)  # 初始化数据集
	clf.fit(train_data_set)  # 训练决策树模型
	print clf.decision_tree  # 输出决策树
	predict_class_label_list = clf.predict(test_data_set)  # 对测试数据集分类
	right_class_label = 0
	for i in range(len(test_data_set)):
		if predict_class_label_list[i] == true_class_label_list[i]:
			right_class_label += 1
	print right_class_label / (len(test_data_set) * 1.0)


if __name__ == '__main__':
	main('train.txt', 'test.txt')