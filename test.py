"""
    构造需要的明细数据，实验期次数据+历史数据明细
    该数据集包括用户信息、实验组和对照组的标记、家方电话的记录、续费情况、影响因素（销售水平、城市线级、流量渠道）
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# 模拟用户信息
np.random.seed(42)
num_users = 20000  # 假设有20000个用户

city_line_mapping = {
    'city_A': 'First',  # city_A 一线城市
    'city_B': 'New_First',  # city_B 新一线城市
    'city_C': 'Second',  # city_C 二线城市
    'city_D': 'Third',  # city_D 三线城市
    'city_E': 'First',  # city_A 四线城市
}

user_ids = np.arange(1, num_users + 1)
cities = np.random.choice(['city_A', 'city_B', 'city_C', 'city_D', 'city_E'], num_users)

# 根据城市来确定线级（固定关系）
line_levels = [city_line_mapping[city] for city in cities]
# 流量渠道设置
main_channels = ['Advertisement', 'Owned_Traffic', 'Referral', 'Partnership']
secondary_channels = ['Email_Marketing', 'SMS_Marketing', 'Offline_Events', 'Other']

# 流量渠道的分配规则可以按照城市的不同来调整
def get_traffic_channel(city):
    if city == 'city_A':
        return np.random.choice(main_channels, p=[0.4, 0.3, 0.2, 0.1])
    elif city == 'city_B':
        return np.random.choice(main_channels, p=[0.3, 0.4, 0.2, 0.1])
    else:
        return np.random.choice(main_channels + secondary_channels, p=[0.2] * 4 + [0.05] * 4)

# 为每个用户分配流量渠道
channels = [get_traffic_channel(city) for city in cities]

# 创建用户数据的DataFrame
users_df = pd.DataFrame({
    'user_id': user_ids,
    'city': cities,
    'line_level': line_levels,
    'traffic_channel': channels
})

# 显示前五行数据
# print(users_df[['user_id', 'city', 'line_level', 'traffic_channel']].head())


# 模拟实验数据
# 假设每个用户都处于不同的实验期次
experiment_phases = np.random.choice([1, 2], num_users)  # 实验期次（1期、2期）
received_call = np.random.choice([0, 1], num_users)  # 是否接到家访电话（0-未接，1-接到）
renewal_status = np.random.choice([0, 1], num_users, p=[0.3, 0.7])  # 续费状态（0-未续，1-已续费）

# 模拟实验期次数据的DataFrame
experiment_df = pd.DataFrame({
    'user_id': user_ids,
    'experiment_phase': experiment_phases,
    'received_call': received_call,
    'renewal_status': renewal_status
})

teachers_data = pd.DataFrame({
    'teacher_id': np.arange(1, num_users + 1),
    'group': np.random.choice(['A', 'B'], num_users)  # A 为实验组，B 为对照组
})

# 为每个用户分配教师ID（假设每个用户都有对应的教师）
users_df['teacher_id'] = np.random.choice(teachers_data['teacher_id'], num_users)

# 合并三个数据集
merged_data = pd.merge(users_df, experiment_df, on='user_id', how='left')
merged_data = pd.merge(merged_data, teachers_data, on='teacher_id', how='left')

# 显示前五行数据
print(merged_data.head())

# 实验组与对照组的用户分布
group_renewal = merged_data.groupby(['group', 'received_call'])['renewal_status'].mean().reset_index()

# 可视化实验组与对照组的续费率对比
sns.barplot(x='group', y='renewal_status', hue='received_call', data=group_renewal)
plt.title('Renewal Rate Comparison by Group and Call Status')
plt.ylabel('Renewal Rate')
plt.show()

# 实验组与对照组的续费率
experimental_group = merged_data[merged_data['group'] == 'A']
control_group = merged_data[merged_data['group'] == 'B']

# 计算实验组与对照组的续费率
exp_renewal_rate = experimental_group['renewal_status'].mean()
ctrl_renewal_rate = control_group['renewal_status'].mean()

# 独立样本t检验
t_stat, p_val = stats.ttest_ind(experimental_group['renewal_status'], control_group['renewal_status'])

# 输出结果
print(f"Experiment group renewal rate: {exp_renewal_rate}")
print(f"Control group renewal rate: {ctrl_renewal_rate}")
print(f"p-value from t-test: {p_val}")

# 如果p值小于显著性水平（通常为0.05），说明两组之间存在显著差异

# 按城市、实验组和是否接到家访电话的交叉分析续费率
group_city_traffic_renewal = merged_data.groupby(['group', 'city', 'traffic_channel', 'received_call'])['renewal_status'].mean().reset_index()


pivot_data = group_city_traffic_renewal.pivot_table(index=['city', 'traffic_channel'], columns='received_call', values='renewal_status')

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, annot=True, cmap='coolwarm', cbar_kws={'label': 'Renewal Rate'}, linewidths=0.5)
plt.title('Renewal Rate by City, Traffic Channel and Received Call')
plt.ylabel('City and Traffic Channel')
plt.show()

# 假设实验期次数据包含时间戳
# 可以通过实验期次及日期来计算时间维度上的续费率

# 模拟日期数据
merged_data['experiment_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(merged_data['experiment_phase']*10, unit='D')

# 按日期和组别计算续费率
time_group_renewal = merged_data.groupby(['experiment_date', 'group'])['renewal_status'].mean().reset_index()

# 绘制实验组与对照组的续费率变化曲线
# plt.figure(figsize=(12, 6))
sns.lineplot(data=time_group_renewal, x='experiment_date', y='renewal_status', hue='group', marker='o')
plt.title('Renewal Rate Over Time by Group')
plt.xlabel('Date')
plt.ylabel('Renewal Rate')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# 绘制密度图（KDE）来查看不同实验组的续费率分布
# plt.figure(figsize=(12, 6))
sns.kdeplot(data=merged_data[merged_data['group'] == 'A'], x='renewal_status', label='Experimental Group', shade=True)
sns.kdeplot(data=merged_data[merged_data['group'] == 'B'], x='renewal_status', label='Control Group', shade=True)
plt.title('Density Plot of Renewal Rate by Group')
plt.xlabel('Renewal Rate')
plt.ylabel('Density')
plt.legend(title='Group')
plt.show()

# 按照组别（实验组和对照组）计算续费率
group_renewal_rate = merged_data.groupby('group')['renewal_status'].mean().reset_index()

# 绘制续费率对比图
plt.figure(figsize=(8, 5))
sns.barplot(data=group_renewal_rate, x='group', y='renewal_status', palette='coolwarm')
plt.title('Comparison of Renewal Rates by Group (With and Without Call)')
plt.xlabel('Group (Experimental vs Control)')
plt.ylabel('Average Renewal Rate')
plt.tight_layout()
plt.show()

# 计算接到家访电话与未接到家访电话的续费率差异
call_impact_on_renewal = merged_data.groupby('received_call')['renewal_status'].mean().reset_index()

# 绘制图表
plt.figure(figsize=(8, 5))
sns.barplot(data=call_impact_on_renewal, x='received_call', y='renewal_status', palette='Set2')
plt.title('Impact of Receiving a Call on Renewal Rate')
plt.xlabel('Received Call (0 = No, 1 = Yes)')
plt.ylabel('Average Renewal Rate')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()




