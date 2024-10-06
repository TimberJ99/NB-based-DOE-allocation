# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:26:15 2022
23.3.16: 
    1. 修正expriment中计算DSO_cost的方法，由考虑DOE分配后用户injection的网络结果导出P0；
    2. 修正expriment中计算用户在分配DOE后的optimal exp和obj，使用solve_in_RDOE方法；
    3. 修正expriment中，不同obj_type对应remove_constr的真假值不同；
    4. 修正indices计算中，算用户在分配DOE后的optimal exp和obj，使用solve_in_RDOE方法；
23.4.21： 
    1. 在NB方法中,remove constr，避免未知原因造成的求解时间过长
23.5.9:
    1. 在test_obj中gather_bidding的时候，需要输入agents_Pmax,为用户光伏出力+最大电池放电功率，并且在NB方法中改变
23.10.12:
    calc_NU里改成使用实际的P_0计算
23.10.28:
    save函数加增加了保存optimal运行下的网络状态
    在fairness指标计算处保存了fairness_agg
23.11.21:
    1. 修改了agent_opt_info生成函数，将生成的crs保存在本目录下的crs文件夹中,同时也修改了对应的m文件
    2. 保存了Index_df
    3. fairness_agg的计算函数中都新增了base=0.x
    4. 保存ed_series
23.12.2：
    1. 改用agent_fl模型
    2. 生成边界方法确定agent_bd_df
    3. test_obj中改用Introduce_DOE
    4. 所有模型求解时,remove_constr都改为False（因为没有使用Gatherbidding方法，所以没有这些constr生成）
    5. agents_Pmax也改了计算方法
23.12.12:
    1. 修改了'energy_exp_ratio'以及'utilization_limit'的计算，np.maximum(current_exp,0),以及'normalized_limit'
    2. 修改了agent_Pmax的计算方式
23.1.31：
    1. 新增计算fairness_new，用时段平均值计算，而不是先逐时段计算再取平均
@author: TimberJ99
"""
import pandas as pd
import numpy as np
from utilis4biddingRW_single_interval_normalizedNB import Args
import pypsa
from gurobipy import *
from scipy.io import loadmat, savemat
from utilis import Show_voltage, Show_trade
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
# from agents import agent, run, epoch
np.random.seed(62)
#%% Import data
all_ed = np.load('./all_ed_141.npy')
price_buy = loadmat('./price_buy.mat')
TOU = price_buy['jg'].flatten()[0::4]
FIT = TOU/2
lambda_0 = 0.6*TOU
T=24

Nodes = pd.read_excel('./141_system.xls', sheet_name='141_system_datas',
                     usecols=list(range(1, 14)),
                     index_col=0,
                     header=3,
                     nrows=141
                     )

Lines = pd.read_excel('./141_system.xls', sheet_name='141_system_datas',
                     usecols=list(range(1, 14)),
                     # index_col=0,
                     header=3,
                     skiprows=145
                     ) # tbus和fbus并未做修改
Lines['rateA'] = 10

load_variation_curve = pd.read_excel('./141_system.xls', sheet_name='Load_data',
                                     usecols=list(range(1,26)),
                                     header=3,
                                     index_col=0,
                                     nrows=2)
ESS = pd.read_excel('./141_system.xls', sheet_name='DER_parameters',
                    usecols=list(range(2,10)),
                    # header=
                    skiprows=21,
                    # dtype=float
                    )[:4].astype(float)
RES = pd.read_excel('./141_system.xls', sheet_name='DER_parameters',
                    usecols=list(range(2,26)),
                    # header=
                    skiprows=29,
                    )[:4].astype(float)
RES.columns = range(24)
num_prosumers = 28
prosumer_lst = np.sort(np.random.choice(np.where(Nodes['Pd']!=0)[0]+1, size=num_prosumers, replace=False)).tolist()
# prosumer_lst = all_ed[::141//28][1:]
# prosumer_lst = [2,33,34,3,4,5,6,37,35,7,88,111,89,38,36,8,90,96,9,53,91,39,101,
#                 108,95,109,13,14]
# prosumer_lst[:5] = [2, 33, 34, 3, 4]

RES = pd.concat([RES]*7)
ESS = pd.concat([ESS]*7)
RES.index = prosumer_lst
ESS.index= prosumer_lst


Pd_dic = {node:Nodes.loc[node, 'Pd'] * load_variation_curve.loc['Active power',:] for node in Nodes.index}
Pd_df = pd.DataFrame(Pd_dic).T
Pd_df.columns = range(T)
Qd_dic = {node:Nodes.loc[node, 'Qd'] * load_variation_curve.loc['Reactive power',:] for node in Nodes.index}
Qd_df = pd.DataFrame(Qd_dic).T
Qd_df.columns = range(T)

for prosumer in prosumer_lst:
    ratio = Pd_df.loc[prosumer,:].sum() / RES.loc[prosumer,:].sum()*2.5*2 # 第一版1.7
    RES.loc[prosumer,:] = ratio * RES.loc[prosumer,:]
    ESS.loc[prosumer, ['Initial energy (MWh)', 'Maximum regulation capacity (MW)',
           'Rated capacity (MWh)', 'Minimum/maximum input/output ']] = ratio * ESS.loc[prosumer, ['Initial energy (MWh)', 'Maximum regulation capacity (MW)',
                  'Rated capacity (MWh)', 'Minimum/maximum input/output ']]



RES.T.plot(legend=False)

RES.loc[:,9:12] *= 0.9
RES.loc[:,12:17] *= 1.2
#%% Generate prosumers
T=24

# T_on, T_off = 8,13
T_on, T_off = 7,16
# T_on, T_off = 1,7
# T_on, T_off = 0,23
T_gap = T_off - T_on + 1 
target = 8

# T_on, T_off = 8, 8
# T_gap = T_off - T_on + 1 
# target = 8

v_nom=12.5
S_base = 10
Z_base = 12.5**2 / 10

            


args = Args(Nodes=Nodes, Lines=Lines, 
            Pd_df=Pd_df, 
            Qd_df=Qd_df, 
            v_min=0.9, 
            v_max=1.05, 
            v_nom=v_nom, 
            S_base=S_base,
            TOU = TOU.flatten(),
            FIT = FIT.flatten(),
            lambda_0 = lambda_0,
            T_on = T_on,
            T_off = T_off,
            T = T_gap
            )

from utilis4biddingRW_single_interval_normalizedNB import generate_agents, gen_mpcoeff
from utilis4biddingRW_single_interval_normalizedNB import agent_fl

# gen prosumers
num_prosumers = 20
prosumer_lst = np.random.choice(prosumer_lst, num_prosumers, replace=False) # 不重复取出

def generate_agents(prosumer_lst, RES, Pd_df):
    '''
    Generate agents' objects

    Parameters
    ----------
    prosumer_lst : List
        prosumers' ids.
    ESS : pandas.DataFrame
        DESCRIPTION.
    RES : pandas.DataFrame
        DESCRIPTION.
    Pd_df : pandas.DataFrame
        The shape should be (num_nodes, T).
        This arg can be obtained from net args.

    Returns
    -------
    agent_dic : Set
        Keys: agent id.
        Values: agent object.

    '''
    agent_dic = {}
    for i, prosumer_id in enumerate(prosumer_lst):
        agent_dic[prosumer_id] = agent_fl(prosumer_id = prosumer_id,
                                       d_min = Pd_df.loc[prosumer_id]*0.2,
                                       d_max = Pd_df.loc[prosumer_id]*2,
                                       P_pv = RES.loc[prosumer_id]*2,
                                        # alpha=0.24*Pd_df.loc[prosumer_id].mean()*0.5,
                                        alpha=0.24,
                                       beta=0.24,
                                       )
    return agent_dic
agent_dic = generate_agents(prosumer_lst, RES.iloc[:,T_on:T_off+1], Pd_df.iloc[:,T_on:T_off+1]) 


# 手动去除agent86，改为agent77
from copy import deepcopy
agent_dic[86] = deepcopy(agent_dic[77])
agent_dic[86].id = 86
agent_dic[86].d_min.name=86
agent_dic[86].d_max.name=86
agent_dic[86].P_pv.name=86
# build prosumers model
for agent in agent_dic.values():
    agent.build_model(args) 
#%% Generate DSO model and solve
def test_obj(DSO_model, obj_type='vf_sum', remove_constr=False, **kwargs):
    '''

    Parameters
    ----------
    DSO_model : TYPE
        DESCRIPTION.
    obj_type : TYPE, optional
        DESCRIPTION. The default is 'vf_sum'.
    **kwargs : TYPE
        Model solving params.

    Returns
    -------
    None.

    '''

    M_ = DSO_model.introduce_DOE(agents_Pmax)
    # DSO_model.calculate_header_limit()
    DSO_model.build_model()
    DSO_model.set_obj(obj_type)
    DSO_model.solve_model(remove_constr, **kwargs)

#%% get back x* and vf*
def test_opt(agent_id,agent_dic, DOE_df, args):
    '''By running agent model in which the DOE is fixed to get optimal opertaion'''
    DOE = DOE_df.loc[:,agent_id]
    # print(f'DOE:{DOE}')
    agent = agent_dic[agent_id]
    agent.build_model(args)
    for t in range(agent.T_on, agent.T_off+1):
        agent.m.addConstr(agent.P_inj[t]<=DOE[t], name=f'testDOEconstr_{t}')
    agent.solve_model()
    # agent.m.write(f'./agent{agent_id}_model.lp') # 写入加入DOE后用户模型
    # print(agent.m.getConstrs())
    
    return agent.get_exp(), agent.m.getObjective().getValue()

def retrieve_opt_cr(agent_dic, DOE_df, args):
    '''get optimal opertaion(used for cr method)'''
    prosumer_opt = {}
    prosumer_exp = {}
    prosumer_obj = {}
    for agent in agent_dic.values():
        print(f'Sovling for agent No.{agent.id}')
        theta = DOE_df.loc[:,agent.id].values.reshape(-1,1)
        x_star = agent.ans.evaluate(theta)
        if x_star is None:
            print('Problem happens.',agent.id)
            p_exp_star, obj_star = test_opt(agent.id, agent_dic, DOE_df, args)
            prosumer_exp[agent.id], prosumer_obj[agent.id] = p_exp_star, obj_star
            continue
        obj_star = agent.ans.evaluate_objective(theta).item()
        
        P_exp_star = x_star[-agent.T:].flatten()
        
        prosumer_exp[agent.id], prosumer_obj[agent.id], prosumer_opt[agent.id] = P_exp_star, obj_star, x_star
    return prosumer_exp, prosumer_obj, prosumer_opt

def retrieve_opt_noncr(agent_dic, DOE_df, args):
    '''get optimal opertaion(used for non-cr method), no use for prosumer_opt'''
    prosumer_opt = {}
    prosumer_exp = {}
    prosumer_obj = {}
    for agent in agent_dic.values():
        p_exp_star, obj_star = test_opt(agent.id, agent_dic, DOE_df, args)
        prosumer_exp[agent.id], prosumer_obj[agent.id] = p_exp_star, obj_star
    return prosumer_exp, prosumer_obj, prosumer_opt
#%% 生成边界
from itertools import product
agent_bd_df = pd.DataFrame(index = product(prosumer_lst,['lb','ub']), columns = range(T_on, T_off+1))
agent_bd_df = agent_bd_df.set_index(pd.MultiIndex.from_tuples(agent_bd_df.index, names=['id', 'bd']))
for i in prosumer_lst:
    for t in range(T_on, T_off+1):
        agent_bd_df.loc[(i,'lb'),t] = max((agent_dic[i].P_pv[t] - agent_dic[i].d_max[t] - agent_dic[i].P_pv[t]*0.6)/S_base,0)
        agent_bd_df.loc[(i,'ub'),t] = max((agent_dic[i].P_pv[t] - agent_dic[i].d_min[t])/S_base,0)
#%% 计算线性化模型的DOE结果
from utilis4biddingRW_single_interval_normalizedNB import DSO_Lin, grid
prosumers_exp = {i:None for i in agent_dic.keys()}
for agent in agent_dic.values():
    # prosumers_exp[agent.id] = (agent.P_pv-agent.P_d)/S_base
    prosumers_exp[agent.id] = 0
init_grid = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumers_exp)
init_grid.build_model()
init_grid.solve_model()
# init_grid = grid(args=args)
init_grid.grid_profile()

agents_Pmax = pd.DataFrame(index=prosumer_lst, columns=range(T_on, T_off+1))
for col_name in agents_Pmax.columns:
    agents_Pmax.loc[:,col_name] = list(map(lambda x: x.P_pv.max()/S_base, agent_dic.values()))
    # agents_Pmax.loc[:,col_name] = list(map(lambda x: x.P_pv.max()*1.5/S_base + x.d_min.min()/S_base, agent_dic.values()))
    # agents_Pmax.loc[:,col_name] = 1

def expriment(args, prosumer_lst, agent_dic, obj_type, **kwargs):
    '''这里只考虑多时段的DOE分配'''
    print('---------------------------------------------------------------------------------------------')
    # 生成DSO模型并根据目标函数优化DOE allocation
    DSO_model = DSO_Lin(args=args, prosumer_lst=prosumer_lst, grid=init_grid)
    
    # 只考虑分配DOE的话则忽略用户可行域需求
    if obj_type in ['log_DOE_sum', 'DOE_sqrt_sum', 'DOE_sum']:
        test_obj(DSO_model, obj_type=obj_type, remove_constr=False, **kwargs)
    else:
        test_obj(DSO_model, obj_type=obj_type, remove_constr=False, **kwargs)
   
    DOE_allocation = DSO_model.allocate_DOE()
    DOE_df = pd.DataFrame.from_dict(DOE_allocation, orient='index').T
    print(DOE_df)
    
    # 绘制DOE allocation
    pd.DataFrame.from_dict(DOE_allocation, orient='index').T.plot(legend=False) 
    ax = plt.gca()
    ax.set_title(f'DOE allocation of {obj_type} by LinearazationDistFlow')
    ax.set_xlabel('Time intervals')
    ax.set_ylabel('DOE allocation')    
    
    # 计算用户在该DOE下的inj和cost
    prosumer_exp, prosumer_obj = {}, {}
    for agent in agent_dic.values():
        prosumer_exp[agent.id], prosumer_obj[agent.id] = agent.solve_in_RDOE(DOE_df, args)
    
    prosumer_cost = sum(prosumer_obj.values())* args.S_base

    # 计算该DOE的用户inj下的DSO购电成本
    net15 = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumer_exp)
    net15.build_model()
    net15.solve_model()
    P_0 = net15.net.generators_t['p']['root'].values
    DSO_cost = (args.lambda_0[args.T_on: args.T_off+1] * P_0).sum()

    
    cost_total = prosumer_cost + DSO_cost
    
    print(f'{obj_type} result: total cost = {cost_total}\n',f'prosumer cost = {prosumer_cost}\n', f'DSO_cost = {DSO_cost}')
    
    res = {}
    res_lst = ['DOE_df', 'prosumer_exp', 'prosumer_obj', 'DSO_cost', 'prosumer_cost', 'cost_total']
    for res_name in res_lst:
        res[res_name] = eval(res_name)
    
    return  DSO_model, res

models = {}
res = {}
obj_lst = [
            'DOE_sum', 
            'log_DOE_sum',
            # 'DOE_sqr_sum',
 
            # 'surplus_sum',
            # 'log_surplus_sum',
            # 'log_PD_sum',

            # 'vf_sum',
            # 'log_vf_sum',

           ]    
for obj_type in obj_lst:
    models[obj_type], res[obj_type] = expriment(
                                                args=args, 
                                                prosumer_lst=prosumer_lst, 
                                                agent_dic=agent_dic, 
                                                obj_type=obj_type,
                                                NoRelHeurTime = 10,
                                                MIPFocus=1,
                                                NumericFocus=3,
                                                TimeLimit=600,
                                                # TimeLimit=60,
                                                NodefileStart=0.5,
                                                # MIPGap=1e-3
                                                )
    # models[obj_type].m.dispose()

# game way

def expriment_NB(args, prosumer_lst, agent_dic, obj_type, **kwargs):
    '''这里只考虑多时段的DOE分配'''
    print('---------------------------------------------------------------------------------------------')
    # 生成DSO模型并根据目标函数优化DOE allocation
    DSO_model = DSO_Lin(args=args, prosumer_lst=prosumer_lst, grid=init_grid)
    
    M_ = DSO_model.introduce_DOE(agents_Pmax)
    DSO_model.build_model()
    if obj_type=='NB':
        DSO_model.build_NB_solution(agent_bd_df, RKS=False)
    elif obj_type=='RKS-NB':
        DSO_model.build_NB_solution(agent_bd_df, RKS=True)
    # DSO_model.set_obj(obj_type)
    DSO_model.solve_model(remove_constr=False, **kwargs)
        
    DOE_allocation = DSO_model.allocate_DOE()
    DOE_df = pd.DataFrame.from_dict(DOE_allocation, orient='index').T
    print(DOE_df)
    
    # 绘制DOE allocation
    pd.DataFrame.from_dict(DOE_allocation, orient='index').T.plot(legend=False) 
    ax = plt.gca()
    ax.set_title(f'DOE allocation of {obj_type} by LinearazationDistFlow')
    ax.set_xlabel('Time intervals')
    ax.set_ylabel('DOE allocation')    
    
    # 计算用户在该DOE下的inj和cost
    prosumer_exp, prosumer_obj = {}, {}
    for agent in agent_dic.values():
        prosumer_exp[agent.id], prosumer_obj[agent.id] = agent.solve_in_RDOE(DOE_df, args)
    
    prosumer_cost = sum(prosumer_obj.values())* args.S_base

    # 计算该DOE的用户inj下的DSO购电成本
    net15 = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumer_exp)
    net15.build_model()
    net15.solve_model()
    P_0 = net15.net.generators_t['p']['root'].values
    DSO_cost = (args.lambda_0[args.T_on: args.T_off+1] * P_0).sum()

    
    cost_total = prosumer_cost + DSO_cost
    
    print(f'{obj_type} result: total cost = {cost_total}\n',f'prosumer cost = {prosumer_cost}\n', f'DSO_cost = {DSO_cost}')
    
    res = {}
    res_lst = ['DOE_df', 'prosumer_exp', 'prosumer_obj', 'DSO_cost', 'prosumer_cost', 'cost_total']
    for res_name in res_lst:
        res[res_name] = eval(res_name)
    
    return  DSO_model, res

for obj_type in [
                'NB', 
                 'RKS-NB'
                 ]:
    models[obj_type], res[obj_type] = expriment_NB(
                                                args=args, 
                                                prosumer_lst=prosumer_lst, 
                                                agent_dic=agent_dic, 
                                                obj_type=obj_type,
                                                NoRelHeurTime = 10,
                                                MIPFocus=1,
                                                NumericFocus=3,
                                                TimeLimit=3600,
                                                NodefileStart=0.5)
    
obj_lst = obj_lst+[
                'NB', 
                   'RKS-NB']
#%% Grid condition of optimal operation
from utilis4biddingRW_single_interval_normalizedNB import grid

def grid_simulate(args, prosumer_lst, prosumer_exp, obj_type=None):
    network = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumer_exp)
    network.build_model()
    network.solve_model()
    
    network.grid_profile()
    network.v.plot(legend=False)
    # network.f_p.plot(legend=False)
    
    # 获得网络状况
    P_0 = network.net.generators_t['p']['root'].values
    DSO_buy = (args.lambda_0[args.T_on: args.T_off+1] * P_0).sum()
    DSO_buy_arr = args.lambda_0[args.T_on: args.T_off+1] * P_0
    loss_profile =  network.net.lines_t.p1.values -  network.net.lines_t.p0.values*(-1)
    loss_arr = loss_profile.sum(axis=1)
    loss = loss_arr.sum()

    f_p = network.net.lines_t.p0
    f_q = network.net.lines_t.q0
    s = np.sqrt(f_p**2 + f_q**2)
    s_percent = s/Lines.loc[0,'rateA']
    
    if obj_type:
        ax = plt.gca()
        ax.set_title(obj_type)
        ax.set_xlabel('Time intervals')
        ax.set_ylabel('Voltage')
        
        grid_simulation = {}
        var_lst = ['P_0', 'DSO_buy_arr', 'loss_arr', 's', 's_percent']
        for var in var_lst:
            grid_simulation[var] = eval(var)
        
        return network, grid_simulation
  
network_res, simulation_res = {}, {}   
for obj_type in obj_lst:
    network_res[obj_type], simulation_res[obj_type] = grid_simulate(args, prosumer_lst, 
                                             prosumer_exp = res[obj_type]['prosumer_exp'], 
                                             obj_type=obj_type)    
#%%% optimal operation
# agent_dic = generate_agents(prosumer_lst, ESS, RES.iloc[:,:T], Pd_df.iloc[:,:T])
prosumer_exp={}

for agent in agent_dic.values():
    agent.build_model(args)
    agent.solve_model()
    # attach self-optimal operation
    agent.opt_exp = agent.get_exp()
    agent.opt_obj = agent.obj.getValue()
    prosumer_exp[agent.id] = agent.get_exp()

# prosumers_exp = {i:None for i in agent_dic.keys()}
# for agent in agent_dic.values():
#     prosumers_exp[agent.id] = (agent.P_pv-agent.P_d)/S_base

from utilis4biddingRW_single_interval_normalizedNB import grid
net15_1 = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumer_exp)
net15_1.build_model()
net15_1.solve_model()
net15_1.grid_profile()
net15_1.v.plot(legend=False)

# 获得网络状况
P_0 = net15_1.net.generators_t['p']['root'].values
DSO_buy = (args.lambda_0[args.T_on: args.T_off+1] * P_0).sum()
DSO_buy_arr = args.lambda_0[args.T_on: args.T_off+1] * P_0
loss_profile =  net15_1.net.lines_t.p1.values -  net15_1.net.lines_t.p0.values*(-1)
loss_arr = loss_profile.sum(axis=1)
loss = loss_arr.sum()

f_p = net15_1.net.lines_t.p0
f_q = net15_1.net.lines_t.q0
s = np.sqrt(f_p**2 + f_q**2)
s_percent = s/Lines.loc[0,'rateA']
#%% save result
import os
storage_dir = f'./Res4DOEallocation/24_1_31_Lin_grid141_T_{T_on}-{T_off}_prosumers{num_prosumers}'
os.makedirs(storage_dir, exist_ok=True)

def save_path(params=None):
    if params is not None:
        storage_path = os.path.join(storage_dir, params)
    else:
        storage_path = os.path.join(storage_dir)
    os.makedirs(storage_path, exist_ok=True)
    return storage_path

# 保存实验结果
def save_result(storage_path):
    np.save(os.path.join(storage_path,'res'), res)
    np.save(os.path.join(storage_path,'simulation_res'), simulation_res)
    
    for obj_type, network in network_res.items():
        # net_stroage_path = './Res4DOEallocation/23_1_1' + f'/{obj_type}'
        net_stroage_path = os.path.join(storage_path, 'network', obj_type)
        os.makedirs(net_stroage_path, exist_ok=True)
        network.net.export_to_csv_folder(net_stroage_path)
        
    # 存储optimal operation下网络结果
    netopt_stroage_path = os.path.join(storage_path, 'network', 'opt')
    os.makedirs(netopt_stroage_path, exist_ok=True)
    net15_1.net.export_to_csv_folder(netopt_stroage_path)
    
    
save_result(save_path())
    
# #%% test gap (run model)
# obj_type = 'DOE_sum'
# DSO_model = DSO_Lin(args, prosumer_lst, T, grid=init_grid)
# test_obj(DSO_model, obj_type=obj_type)
# DOE_allocation = DSO_model.allocate_DOE()
# DOE_df = pd.DataFrame.from_dict(DOE_allocation, orient='index').T
# print(DOE_df)

# # 绘制DOE allocation
# pd.DataFrame.from_dict(DOE_allocation, orient='index').T.plot(legend=False) 
# ax = plt.gca()
# ax.set_title(f'DOE allocation of {obj_type}')
# ax.set_xlabel('Time intervals')
# ax.set_ylabel('DOE allocation')    

# # if obj_type=='vf_sum':
# #     prosumer_exp, prosumer_obj, _ = retrieve_opt_cr(agent_dic, DOE_df, T)
# # else:
# prosumer_exp, prosumer_obj, _ = retrieve_opt_cr(agent_dic, DOE_df, T)

# prosumer_cost = sum(prosumer_obj.values())* args.S_base

# DSO_cost = sum(DSO_model.P_0[t].x * lambda_0[t] * args.S_base for t in range(T))

# cost_total = prosumer_cost + DSO_cost

# print(f'{obj_type} result: total cost = {cost_total}\n',f'prosumer cost = {prosumer_cost}\n', f'DSO_cost = {DSO_cost}')

# v = {node:{} for node in range(1,141+1)}    
# for (node, t), var in DSO_model.v.items():
#     v[node][t] = np.sqrt(var.x)
    
# pd.DataFrame.from_dict(v, orient='index').T.plot(legend=False)

# v_s = network_res['DOE_sum'].v
# v_s.plot(legend=False)
# inj=network_res['DOE_sum'].net.buses_t.p

# # DSO_model.P_0
# #%% test gap (use model res)
# DSO_model = models['DOE_sum']
# f_p = {k:v.x for k,v in DSO_model.f_p.items()}
# f_q = {k:v.x for k,v in DSO_model.f_q.items()}
# v_squared = {k:v.x for k,v in DSO_model.v.items()}
# I_squared = {k:v.x for k,v in DSO_model.l.items()}
# gap={}

# for k,v in f_p.items():
#     gap[k] = abs(f_p[k]**2 + f_q[k]**2 - v_squared[k]*I_squared[k])
    


# gap_max = max(gap.keys(), key=(lambda k: gap[k]))

# for i in range(10):
#     gap_max = sorted(gap, key=gap.get, reverse=True)[i]
#     print(f'gap max{i} is {gap[gap_max]}, happens at {gap_max}')
 
# #%%  test Linres
# obj_type = 'DOE_sum'
# # cr 跑出来的v
# DSO_model = models[obj_type]
# v = {node:{} for node in range(1,141+1)}    
# for (node, t), var in DSO_model.v.items():
#     # v[node][t] = np.sqrt(var.x)
#     v[node][t] = var.x
    
# pd.DataFrame.from_dict(v, orient='index').T.plot(legend=False)

# # 仿真的v (真实exp)
# # v_s = network_res[obj_type].v
# # v_s.plot(legend=False)


# # inj=network_res['DOE_sum'].net.buses_t.p

# # 仿真的v （DOE注入）
# grid_simulate(T, args, prosumer_lst, prosumer_exp=res[obj_type]['DOE_df'])
# #%% 使用DSO模型的LinDistFlow跑PF
# obj_type = 'DOE_sum'
# for t in range(T):
#     for i in prosumer_lst:
#         DSO_model.m.addConstr(DSO_model.DOE[i,t] == res[obj_type]['DOE_df'].loc[t,i])
# DSO_model.m.update()
# DSO_model.m.setObjective(0)
# DSO_model.solve_model()

# v = {node:{} for node in range(1,141+1)}    
# for (node, t), var in DSO_model.v.items():
#     # v[node][t] = np.sqrt(var.x)
#     v[node][t] = var.x
    
# pd.DataFrame.from_dict(v, orient='index').T.plot(legend=False)
# #%% 使用额外DistFlow测试,初始点随机
# import copy
# from utilis4bidding import DSO_Lin, grid

# prosumers_exp = {i:None for i in agent_dic.keys()}
# for agent in agent_dic.values():
#     prosumers_exp[agent.id] = (agent.P_pv-agent.P_d)/S_base
# init_grid = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumers_exp)

# from DistFlow import LinDistFlow
# args_PF = copy.deepcopy(args)
# Pd_df_PF = args_PF.Pd_df
# for i in prosumer_lst:
#     Pd_df_PF.loc[i,:(T-1)] = res[obj_type]['DOE_df'].loc[:(T-1), i] * S_base*(-1)
    
# PF_model = LinDistFlow(init_grid, args_PF, T)
# PF_model.build_model()
# PF_model.solve_model()
# # DSO_model.test_gap()
# #%% 使用额外DistFlow测试，初始点用DOE结果
# import copy
# from utilis4bidding import DSO_Lin, grid

# prosumers_exp = {i:None for i in agent_dic.keys()}
# for agent in agent_dic.values():
#     prosumers_exp[agent.id] = (agent.P_pv-agent.P_d)/S_base
# init_grid = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = res[obj_type]['DOE_df'])

# from DistFlow import LinDistFlow
# args_PF = copy.deepcopy(args)
# Pd_df_PF = args_PF.Pd_df
# for i in prosumer_lst:
#     Pd_df_PF.loc[i,:(T-1)] = res[obj_type]['DOE_df'].loc[:(T-1), i] * S_base*(-1)
    
# PF_model = LinDistFlow(init_grid, args_PF, T)
# PF_model.build_model()
# PF_model.solve_model()
# # DSO_model.test_gap()
# v = {node:{} for node in range(1,141+1)}    
# for (node, t), var in PF_model.v.items():
#     # v[node][t] = np.sqrt(var.x)
#     v[node][t] = var.x
    
# pd.DataFrame.from_dict(v, orient='index').T.plot(legend=False, title = 'Lindistflow res')

# init_grid.v.plot(legend=False, title='Simulation res')
#%% check network integrity
for obj_type in obj_lst:
    print('----------------------------------------------')
    print(obj_type)
    simulation_res[obj_type]['loss_total'] = simulation_res[obj_type]['loss_arr'].sum()
    print(f"Total loss is {simulation_res[obj_type]['loss_total']}")
    s_percent = simulation_res[obj_type]['s_percent'].values
    s_percent_overhalf_count = np.where(s_percent>0.5, 1, 0).sum()
    print(f'lines over 0.5 capacity time is {s_percent_overhalf_count}')
#%% Indices新

def index_calculation(agent_dic, DOE_df, args):
    index_lst = ['energy_exp_ratio',
                 'energy_exp_diff', 
                 'net_benefit_ratio', 
                 'normalized_limit',
                 'DOE_optexp_diff', 
                 'utilization_limit',
                 # 'DOE_to_minimum',
                 # 'DOE_to_maximum',
                 ]
    index_dic = {index_name:{} for index_name in index_lst}
    # energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic = {}, {}, {}, {}
    # index_lst = [energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic]
    
    for agent in agent_dic.values():
        # get self-optimal opeartion result
        agent.build_model(args)
        agent.solve_model()
        agent.opt_exp = agent.get_exp()
        agent.opt_obj = agent.obj.getValue()
        
        # get the operation under DOE exerted
        # current_exp, current_obj = test_opt(agent.id, agent_dic, DOE_df, args) 
        current_exp, current_obj = agent.solve_in_RDOE(DOE_df, args)
        # print(agent.id,':','current_exp:',current_exp,'current_obj:',current_obj)
        # print(agent.id,':','opt_exp:',agent.opt_exp,'opt_obj:',agent.opt_obj)
        index_dic['energy_exp_ratio'][agent.id] = np.minimum(np.maximum(current_exp / agent.opt_exp,0), 1)
        index_dic['energy_exp_diff'][agent.id] = np.abs(current_exp-agent.opt_exp)
        index_dic['net_benefit_ratio'][agent.id] = -current_obj / -agent.opt_obj  - 1
        index_dic['normalized_limit'][agent.id] = np.minimum(DOE_df.loc[:,agent.id].values / np.maximum(agent.opt_exp,1e-5), 1)
        index_dic['DOE_optexp_diff'][agent.id] = np.abs(DOE_df.loc[:,agent.id].values-agent.opt_exp)
        index_dic['utilization_limit'][agent.id] = np.maximum(current_exp,0) / (DOE_df.loc[:,agent.id].values + 1e-5)
        # energy_exp_ratio = current_exp / agent.opt_exp
        # net_benefit_ratio = -current_obj / -agent.opt_obj  - 1
        # normalized_limit = DOE_df.loc[:,agent.id].values / agent.opt_exp 
        # utilization_limit = (current_exp + 1e-5) / (DOE_df.loc[:,agent.id].values + 1e-5)

        # print('-------------------------------')
        # print(f'Agent: {agent.id}')
        # print(energy_exp_ratio)
        # print(net_benefit_ratio)
        # print(normalized_limit)
        # print(utilization_limit)
        # energy_exp_ratio_dic[agent.id] = energy_exp_ratio
        # net_benefit_ratio_dic[agent.id] = net_benefit_ratio
        # normalized_limit_dic[agent.id] = normalized_limit
        # utilization_limit_dic[agent.id] = utilization_limit
        
        index_df_dic = {}
        for index_name, index in index_dic.items():
            if index_name in ['DOE_optexp_diff','energy_exp_diff']:
                index_df = pd.DataFrame.from_dict(index, orient='index').sum(axis=1)
            index_df = pd.DataFrame.from_dict(index, orient='index').mean(axis=1)
            index_df_dic[index_name] = index_df
        
        
    return index_df_dic

index_res = {}
for obj_type in obj_lst:
    DOE_df = res[obj_type]['DOE_df']
    # print(obj_type)
    index_res[obj_type] = index_calculation(agent_dic, DOE_df, args)
    
index_df_res = pd.DataFrame.from_dict({(i,j): index_res[i][j]
                             for i in index_res.keys()
                             for j in index_res[i].keys()}, orient='index')


index_lst = ['energy_exp_ratio',
             'energy_exp_diff', 
             'net_benefit_ratio', 
             'normalized_limit',
             'DOE_optexp_diff', 
             'utilization_limit',
              # 'DOE_to_minimum',
             # 'DOE_to_maximum',
             ]
for index in index_lst:       
    fig, ax = plt.subplots(figsize=(12,4))
    ax.boxplot([index_df_res.loc[obj_type, index] for obj_type in obj_lst],
               labels=obj_lst,
               showmeans=True)
    ax.set_title(index)
    fig.autofmt_xdate()
    plt.savefig(storage_dir+'/'+f'{index}.png')
    
# index_df_lst0 = index_calculation(agent_dic, res['surplus_sum']['DOE_df'], T)
# index_df_lst1 = index_calculation(agent_dic, res['DOE_sum']['DOE_df'], T)
# index_df_lst2 = index_calculation(agent_dic, res['DOE_sqr_sum']['DOE_df'], T)
        
# energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic = index_calculation(agent_dic, res['surplus_sum']['DOE_df'], T)
# # test_opt(12, agent_dic, res['DOE_sum']['DOE_df'])
# for index_dic in [utilization_limit_dic]:
#     index_agent = pd.DataFrame.from_dict(index_dic).mean(axis=0)
#     print(index_agent)
    
# fig, ax = plt.subplots()
# ax.boxplot(pd.Series(net_benefit_ratio_dic),showmeans=True)

# fig, ax = plt.subplots()
# ax.boxplot([index_df_lst0[1],index_df_lst1[1],index_df_lst2[1]],showmeans=True)
# ax.set_title(r'$\frac{True\,\,surplus}{Intended\,\,surplus} - 1$')

# fig, ax = plt.subplots()
# ax.boxplot([index_df_lst0[0],index_df_lst1[0],index_df_lst2[0]],showmeans=True)
# ax.set_title(r'$\frac{True\,\,Export}{Intended\,\,Export}$')

# fig, ax = plt.subplots()
# ax.boxplot([index_df_lst0[-1],index_df_lst1[-1],index_df_lst2[-1]],showmeans=True)
# ax.set_title(r'$\frac{True\,\,Export}{DOE}$')

for obj_type in obj_lst:
    overmax, belowmin = 0, 0
    for agent in agent_dic.values():
        # if (res[obj_type]['DOE_df'].loc[:,agent.id].values > agent_bd_df.loc[(agent.id, 'ub')].values+1e-5).any():
        #     print(agent.id)
        belowmin += (res[obj_type]['DOE_df'].loc[:,agent.id].values < agent_bd_df.loc[(agent.id, 'lb')].values-1e-5).sum()
        overmax += (res[obj_type]['DOE_df'].loc[:,agent.id].values > agent_bd_df.loc[(agent.id, 'ub')].values+1e-5).sum()
        
    print(f'{obj_type}:overmax={overmax}, belowmin={belowmin}.')    

# 保存
indices=['energy_exp_ratio',
             'net_benefit_ratio', 
             'normalized_limit',
             'utilization_limit']
index_df_res = index_df_res[index_df_res.index.get_level_values(1).isin(indices)]
index_df_res.to_csv(os.path.join(storage_dir, 'index_df_res.csv'))
# #%% Indices旧

# def index_calculation(agent_dic, DOE_df, args):
#     index_lst = ['energy_exp_ratio', 
#                  'net_benefit_ratio', 
#                  'normalized_limit', 
#                  'utilization_limit']
#     index_dic = {index_name:{} for index_name in index_lst}
#     # energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic = {}, {}, {}, {}
#     # index_lst = [energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic]
    
#     for agent in agent_dic.values():
#         # get self-optimal opeartion result
#         agent.build_model(args)
#         agent.solve_model()
#         agent.opt_exp = agent.get_exp()
#         agent.opt_obj = agent.obj.getValue()
        
#         # get the operation under DOE exerted
#         # current_exp, current_obj = test_opt(agent.id, agent_dic, DOE_df, args) 
#         current_exp, current_obj = agent.solve_in_RDOE(DOE_df, args)
        
#         index_dic['energy_exp_ratio'][agent.id] = current_exp / agent.opt_exp
#         index_dic['net_benefit_ratio'][agent.id] = -current_obj / -agent.opt_obj  - 1
#         index_dic['normalized_limit'][agent.id] = DOE_df.loc[:,agent.id].values / agent.opt_exp 
#         index_dic['utilization_limit'][agent.id] = (current_exp + 1e-5) / (DOE_df.loc[:,agent.id].values + 1e-5)
#         # energy_exp_ratio = current_exp / agent.opt_exp
#         # net_benefit_ratio = -current_obj / -agent.opt_obj  - 1
#         # normalized_limit = DOE_df.loc[:,agent.id].values / agent.opt_exp 
#         # utilization_limit = (current_exp + 1e-5) / (DOE_df.loc[:,agent.id].values + 1e-5)

#         # print('-------------------------------')
#         # print(f'Agent: {agent.id}')
#         # print(energy_exp_ratio)
#         # print(net_benefit_ratio)
#         # print(normalized_limit)
#         # print(utilization_limit)
#         # energy_exp_ratio_dic[agent.id] = energy_exp_ratio
#         # net_benefit_ratio_dic[agent.id] = net_benefit_ratio
#         # normalized_limit_dic[agent.id] = normalized_limit
#         # utilization_limit_dic[agent.id] = utilization_limit
        
#         index_df_dic = {}
#         for index_name, index in index_dic.items():
#             index_df = pd.DataFrame.from_dict(index, orient='index').mean(axis=1)
#             index_df_dic[index_name] = index_df
        
        
#     return index_df_dic

# index_res = {}
# for obj_type in obj_lst:
#     DOE_df = res[obj_type]['DOE_df']
    
#     index_res[obj_type] = index_calculation(agent_dic, DOE_df, args)
    
# index_df_res = pd.DataFrame.from_dict({(i,j): index_res[i][j]
#                              for i in index_res.keys()
#                              for j in index_res[i].keys()}, orient='index')


# index_lst = ['energy_exp_ratio', 
#              'net_benefit_ratio', 
#              # 'normalized_limit', 
#              'utilization_limit']
# for index in index_lst:       
#     fig, ax = plt.subplots(figsize=(12,4))
#     ax.boxplot([index_df_res.loc[obj_type, index] for obj_type in obj_lst],
#                labels=obj_lst,
#                showmeans=True)
#     ax.set_title(index)
#     fig.autofmt_xdate()
#     plt.savefig(storage_dir+'/'+f'{index}.png')
    
# # index_df_lst0 = index_calculation(agent_dic, res['surplus_sum']['DOE_df'], T)
# # index_df_lst1 = index_calculation(agent_dic, res['DOE_sum']['DOE_df'], T)
# # index_df_lst2 = index_calculation(agent_dic, res['DOE_sqr_sum']['DOE_df'], T)
        
# # energy_exp_ratio_dic, net_benefit_ratio_dic, normalized_limit_dic, utilization_limit_dic = index_calculation(agent_dic, res['surplus_sum']['DOE_df'], T)
# # # test_opt(12, agent_dic, res['DOE_sum']['DOE_df'])
# # for index_dic in [utilization_limit_dic]:
# #     index_agent = pd.DataFrame.from_dict(index_dic).mean(axis=0)
# #     print(index_agent)
    
# # fig, ax = plt.subplots()
# # ax.boxplot(pd.Series(net_benefit_ratio_dic),showmeans=True)

# # fig, ax = plt.subplots()
# # ax.boxplot([index_df_lst0[1],index_df_lst1[1],index_df_lst2[1]],showmeans=True)
# # ax.set_title(r'$\frac{True\,\,surplus}{Intended\,\,surplus} - 1$')

# # fig, ax = plt.subplots()
# # ax.boxplot([index_df_lst0[0],index_df_lst1[0],index_df_lst2[0]],showmeans=True)
# # ax.set_title(r'$\frac{True\,\,Export}{Intended\,\,Export}$')

# # fig, ax = plt.subplots()
# # ax.boxplot([index_df_lst0[-1],index_df_lst1[-1],index_df_lst2[-1]],showmeans=True)
# # ax.set_title(r'$\frac{True\,\,Export}{DOE}$')
#%% plot surplus
surplus_dic = {obj_type:{'total_surplus':-n, 'DSO_surplus':-j, 'prosumer_surplus':-v} for obj_type in obj_lst
                for m,n in res[obj_type].items() if m == 'cost_total'
               for i,j in res[obj_type].items() if i == 'DSO_cost'
               for k,v in res[obj_type].items() if k == 'prosumer_cost'}
surplus_df = pd.DataFrame.from_dict(surplus_dic, orient='index')

fig, ax = plt.subplots(figsize=(12,4))
x = np.arange(surplus_df.shape[0])
width = 0.3
ax.set_xticks(x)
ax.set_xticklabels(list(surplus_df.index))
ax.bar(x-width/2, surplus_df.iloc[:,0], width=width, label='total_surplus')
ax.bar(x+width/2, surplus_df.iloc[:,1], width=width, label='DSO_surplus')
ax.bar(x+width+width/2, surplus_df.iloc[:,2], width=width, label='prosumer_surplus')
ax.set_title('surplus plot')

fig.autofmt_xdate()
plt.savefig(storage_dir+'/'+'surplus.png')
#%% calculate fairness index
from functools import reduce

from utilis4biddingRW_single_interval_normalizedNB import DSO_Lin, grid
DSO_model = DSO_Lin(args=args, prosumer_lst=prosumer_lst, grid=init_grid)
DSO_model.introduce_DOE(agents_Pmax)
P_tx = DSO_model.calculate_header_limit()
# P_max_agent = pd.DataFrame(list(map(lambda x: x.P_pv/S_base + x.P_batt_max/S_base, agent_dic.values()))).T
P_max_agent = agents_Pmax.T

# def calc_NU(obj_type):
#     models[obj_type].P_tx = P_tx
#     NU = models[obj_type].calculate_NU()
#     return NU

def calc_NU(obj_type):
    # P_ex = res[obj_type]['P_ex']
    P_ex = simulation_res[obj_type]['P_0']/S_base # 修改为实际的P_0
    # models[obj_type].P_tx = P_tx
    NU = P_ex / P_tx
    return np.maximum(NU,0)

def calc_DCU(obj_type):
    DOE = res[obj_type]['DOE_df']
    DOE_sum = DOE.sum(axis=1)
    # total_max_output = sum(map(lambda x: x.P_pv/S_base + x.P_batt_max/S_base, agent_dic.values()))
    total_max_output = agents_Pmax.sum(axis=0)
    # total_max_output = pd.Series(index=list(range(T_on, T_off+1)), data=np.ones(T_off-T_on+1)*0.5*len(prosumer_lst))
    DCU = DOE_sum / total_max_output
    return DCU

def calc_QoS(obj_type):
    DOE = res[obj_type]['DOE_df']
    # P_max_agent = pd.DataFrame(list(map(lambda x: x.P_pv/S_base + x.P_batt_max/S_base, agent_dic.values()))).T
    P_hat = DOE/P_max_agent
    # P_hat = DOE/0.5
    QoS = (P_hat.sum(axis=1))**2/(len(prosumer_lst)*(P_hat**2).sum(axis=1))
    return QoS

def calc_QoE(obj_type):
    DOE = res[obj_type]['DOE_df']
    P_hat = DOE/P_max_agent
    # P_hat = DOE/0.5
    QoE = 1- P_hat.std(axis=1)/0.5
    return QoE

def calc_MMF(obj_type):
    DOE = res[obj_type]['DOE_df']
    P_hat = DOE/P_max_agent
    # P_hat = DOE/0.5
    MMF = P_hat.min(axis=1) / P_hat.max(axis=1)
    return MMF

def fairness_assessment(obj_type):
    NU = calc_NU(obj_type)
    DCU = calc_DCU(obj_type)
    QoS = calc_QoS(obj_type)
    QoE = calc_QoE(obj_type)
    MMF = calc_MMF(obj_type)
    
    fairness_index = pd.concat([NU,DCU,QoS,QoE,MMF], axis=1)
    fairness_index.columns = [
        'NU', 
        'DCU', 
        'QoS', 
        'QoE', 
        'MMF']
    fairness_index.index.name = obj_type
    return fairness_index

fairness_lst = [fairness_assessment(obj_type) for obj_type in obj_lst]
fairness_df = pd.concat(fairness_lst, axis=0, keys=[df.index.name for df in fairness_lst])

# 算所有时段平均值
fairness_agg = fairness_df.groupby(fairness_df.index.get_level_values(0)).mean()
fairness_agg = fairness_agg.reindex(obj_lst)
fairness_agg.to_csv(os.path.join(storage_dir, 'fairness_agg.csv'))

# 用时段平均值计算，而不是先逐时段计算再平均
fairness_new = fairness_agg.copy()
for obj_type in obj_lst:
    DOE = res[obj_type]['DOE_df']
    P_hat = DOE/P_max_agent
    MMF = P_hat.mean(axis=0).min()/P_hat.mean(axis=0).max()
    QoE = 1- P_hat.mean(axis=0).std()/0.5
    QoS = (P_hat.mean(axis=0).sum())**2/(len(prosumer_lst)*(P_hat.mean(axis=0)**2).sum())
    fairness_new.loc[obj_type,'MMF'] = MMF
    fairness_new.loc[obj_type,'QoE'] = QoE
    fairness_new.loc[obj_type,'QoS'] = QoS
    
fairness_new.to_csv(os.path.join(storage_dir, 'fairness_new.csv'))
# 绘图
x = np.arange(len(fairness_agg.index))
num_columns = len(fairness_agg.columns)
width = 0.8 / num_columns  # Adjust the width based on the number of columns

# Create the figure and axis
fig, ax = plt.subplots()

# Plot the bars for each column
for i, column in enumerate(fairness_agg.columns):
    y = fairness_agg[column]
    offset = (i - num_columns / 2) * width  # Offset each bar based on the width and number of columns
    ax.bar(x + offset, y, width, label=column)

# Customize the plot
ax.set_xlabel('Allocation Methods')
ax.set_ylabel('Fairness Index')
ax.legend(bbox_to_anchor=(1.04, 1))
ax.set_xticklabels(['']+list(fairness_agg.index), rotation=45, ha='right')

# Show the plot
plt.savefig(storage_dir+'/'+'fairness.png', bbox_inches='tight')
#%% plot far away prosumers
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
net_graph = net15_1.net.graph(weight='x') # set resistance of lines as edge weight
# nx.get_edge_attributes(net_graph, name='weight')
ed_series = pd.Series(index=prosumer_lst)
for agent in agent_dic.values():
    agent_ed = nx.dijkstra_path_length(net_graph, source='bus_1', target=f'bus_{agent.id}', weight='weight')
    agent.ed = agent_ed
    ed_series[agent.id] = agent.ed

ed_series = ed_series.sort_values()
ed_df = pd.DataFrame(ed_series)
ed_df.to_csv(storage_dir+'/'+'ed_df.csv')


# index_df_sorted = index_df_res.T.loc[ed_series.index]
# for index in index_lst:
#     fig, ax = plt.subplots(figsize=(20,4))
#     for obj_type in obj_lst:
        
#         ax.plot(np.arange(index_df_sorted.shape[0]), index_df_sorted[obj_type, index],
#                 label=obj_type)
#     x = np.arange(index_df_sorted.shape[0])
#     ax.set_xticks(x)
#     ax.set_xticklabels(ed_series.index)
#     ax.legend()
#     ax.set_title(index)
#     fig.autofmt_xdate()
#     # plt.savefig(storage_dir+'/'+index+'+surplus.png')

# # index_df_sorted.plot(y=('DOE_sum', 'net_benefit_ratio'), use_index=False)
# # ax = plt.gca()
# # ax.set_xticklabels(ed_series.index)
# # #%%
# # def isPSD(A, tol=1e-8):
# #   E = np.linalg.eigvalsh(A)
# #   return np.all(E > -tol)


# # for h in models['DOE_sum'].coeff_calculate()['He'].values():
# #     print(isPSD(h))

# 看一看全节点的电气距离
all_ed_series = pd.Series(index=args.Nodes.index)
for node in args.Nodes.index:
    node_ed = nx.dijkstra_path_length(net_graph, source='bus_1', target=f'bus_{node}', weight='weight')
    all_ed_series[node] = node_ed

all_ed_series = all_ed_series.sort_values()

for obj_type in obj_lst:
    fig, ax = plt.subplots()
    DOE = res[obj_type]['DOE_df']
    DOE = DOE.sum(axis=0)
    DOE = DOE[ed_series.index]
    DOE.plot(axes=ax, kind='bar', legend=False)
    
#%% 用平均值计算MMF，而不是先逐时段计算再平均
for obj_type in obj_lst:
    DOE = res[obj_type]['DOE_df']
    P_hat = DOE/P_max_agent
    MMF = P_hat.mean(axis=0).min()/P_hat.mean(axis=0).max()
    QoE = 1- P_hat.mean(axis=0).std()/0.5
    QoS = (P_hat.mean(axis=0).sum())**2/(len(prosumer_lst)*(P_hat.mean(axis=0)**2).sum())
    # print(obj_type,":", MMF)
    # print(obj_type,":", QoE)
    print(obj_type,":", QoS)