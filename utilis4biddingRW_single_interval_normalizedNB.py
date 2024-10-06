# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:30:32 2022
1. 将所有时间参数(T_on, T_off, T)捆绑至args，以便rolling window;
2. 除agent类，DSO,grid均从args中获取build_model所需参数，无需在build_model中输入参数;
3. 将DOE,vf,neg_vf的lb,ub改变适配15节点.

2023.3.1:识别DSO模型中所有value fuction/DOE requirement相关约束，存储到self.prosumers_relevent_constrs中。
在进行CIA求解时，设置self.solve_model函数参数，删除这些约束（不考虑用户最低所需DOE）

2023.3.8:增加gen_DOE_bid函数获得用户需要的ODE上下限；DSO类中增加build_NB_solution方法研究博弈论的DOE分配方法。

2023.3.9:增加construct_feasible_region用来获取所有可行域，但是目前只可以检测双节点单时段
2023.3.14:remove_constr时把y,z,vf全remove
2023.3.19: 去除DSO父类中lines的重复列，也即Lines_reformulated做处理
2023.4.8: gather_bidding中通过agents_opt_info中的cr信息确定各个用户的cr下界
2023.4.20： 修正gather_bidding中cr仅一个时的错误
2023.4.21: 
1. 修正gather bidding中的分段建模方法，采用official方法，并且丢弃最后一个cr，保证vf_sum的DOE分配可以利用DOE_ub
2. 修改get_DOE_bd的方法，不再使用最右的cr作为ub,而是遍历寻找ub
2023.5.9:
gather_bidding增加agents_Pmax输入，更改DOE的ub为用户光伏出力+最大电池放电功率
2023.5.15:
1. gather_bidding中vf_lb和E@DOE<=f均允许小误差，否则会出现DOE最优解不在cr内的情况
2. get_DOE_bd方法中分别算lb和ub，否则ub的break影响lb的结果
3. gather_bidding中vf的ub设置为0.5，以前是0，有一些用户只有成本会vf>0

还没解决：
build_NB_solution中RKS-NB方法，分母中加1e-7，防止出现分分母为0.然后aux2的变量范围要变成正负无穷，因为log里面是无穷大。但是分母为0貌似这个计算方法不成立
gather_bidding中vE@DOE<=f允许小误差，还会出现DOE最优解不在cr内的情况
2023.5.19:
1. 修改用户模型中充放电约束表示方式，换位zero-cost那篇的样子，方便改为rolling window
2023.5.23:
1. solve_in_RDOE(self, DOE_df, args)增加appointed_t参数，用于获得指定t的obj和exp
2. get_DOE_bd增加try函数，帮助判断是否能够计算顶点（cdd库有问题）可能计算不出顶点
2023.10.8:
1. 将RKS-NB中的目标函数修改为了好几种来测试
2. RKS-NB中的aux变量都改成了属性
2023.11.5:
1. 当DOE对于用户不可行时，solve_in_RDOE中设置exp为DOE的值
2023.12.2:
    1. 增加agent_fl模型
    2. 增加introduce_DOE方法，直接引入DOE变量
@author: TimberJ99
"""
import numpy as np
import pandas as pd
from gurobipy import *
import pypsa
import gurobipy_pandas as gppd
from pypoman import compute_polytope_vertices
from copy import deepcopy
#%% generate and solve functions
def generate_agentsRW(prosumer_lst, ESS, RES, Pd_df, SOC_0):
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
        agent_dic[prosumer_id] = agent_opt(prosumer_id = prosumer_id,
                                       SOC_0 = SOC_0[prosumer_id],
                                       SOC_min = 0,
                                       SOC_max = ESS.loc[prosumer_id, 'Rated capacity (MWh)']+2,
                                       P_batt_min = -ESS.loc[prosumer_id, 'Minimum/maximum input/output '],
                                       P_batt_max = ESS.loc[prosumer_id, 'Minimum/maximum input/output '],
                                       eta=0.95,
                                       P_d = Pd_df.loc[prosumer_id],
                                       P_pv = RES.loc[prosumer_id],
                                       # lamda = Lamda_init[prosumer_id],
                                       # local_estimate = global_E_init[prosumer_id],
                                       # threshold = 0,
                                       # Psi = Psi_init[prosumer_id],
                                       # DSO_DOE = DOE_init[prosumer_id]
                                       )
    return agent_dic

def generate_agents(prosumer_lst, ESS, RES, Pd_df):
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
        agent_dic[prosumer_id] = agent_opt(prosumer_id = prosumer_id,
                                       SOC_0 = ESS.loc[prosumer_id, 'Initial energy (MWh)'],
                                       SOC_min = 0,
                                       SOC_max = ESS.loc[prosumer_id, 'Rated capacity (MWh)']+2,
                                       P_batt_min = -ESS.loc[prosumer_id, 'Minimum/maximum input/output '],
                                       P_batt_max = ESS.loc[prosumer_id, 'Minimum/maximum input/output '],
                                       eta=0.95,
                                       P_d = Pd_df.loc[prosumer_id],
                                       P_pv = RES.loc[prosumer_id],
                                       # lamda = Lamda_init[prosumer_id],
                                       # local_estimate = global_E_init[prosumer_id],
                                       # threshold = 0,
                                       # Psi = Psi_init[prosumer_id],
                                       # DSO_DOE = DOE_init[prosumer_id]
                                       )
    return agent_dic

def gen_mpcoeff(agent_dic, args, prosumer_lst=None, T=None):
    S_base = args.S_base
    TOU, FIT = args.TOU, args.FIT
    if prosumer_lst is None:
        prosumer_lst = list(agent_dic.keys())
        for agent in agent_dic.values():
            T = agent.T
            T_on, T_off = agent.T_on, agent.T_off
            break
    
    I = np.eye(T)
    e = np.ones((T,1))
    O = np.zeros(shape=(T,T))
    o = np.zeros(shape=(T,))
    M = np.zeros(shape=(T,T))
    # for i in range(M.shape[0]):
    #     if i <= M.shape[0]-2:
    #         M[i,i]=-1
    #         M[i,i+1]=1
    # M[T-1,0] = 1
    M = np.eye(T) + np.diag(v=-np.ones((T-1,)), k=-1)
    
    agent_mpcoeff = {matrix:{} for matrix in ['A', 'b', 'F', 'Aeq', 'beq', 
                                              'Atheta', 'btheta', 'c','H']}
    for agent_id, agent in agent_dic.items():
        eta = agent.eta
        
        A = np.block([
            [-I,O,O,O,O],
            [O,-I,O,O,O],
            [O,O,-I,O,O],
            [O,O,I,O,O],
            [O,O,O,-I,O],
            [O,O,O,I,O],
            [O,O,O,O,I],
            ])
        
        b = np.hstack([o,o,-agent.P_batt_min/S_base*np.ones((T,)), agent.P_batt_max/S_base*np.ones((T,)),
                       -agent.SOC_min/S_base*np.ones((T,)),agent.SOC_max/S_base*np.ones((T,)), o]).reshape(-1,1)
        
        F = np.vstack([*[O]*6, I])
        
        
        # N = np.pad(-np.eye(T-1)*agent.eta,pad_width=(0,1))
        N = -I
        Aeq = np.block([
            [I, -I, -I, O, O],
            [O, O, N, M, O],
            [I, -I, O, O, I],
            ])

        
        # Aeq[2*T-1] = np.zeros_like(Aeq[2*T-1])
        # Aeq[2*T-1, 3*T]=1
        
        
        beq = np.hstack([agent.P_d/S_base - agent.P_pv/S_base, o, o]).reshape(-1,1)
        # beq[2*T-1,:] = agent.SOC_0/S_base
        beq[T,:] = agent.SOC_0/S_base
        
        Feq = np.zeros(shape=(3*T, T))
        
        Atheta = np.block([
            [-I],
            [I]
            ])
        btheta = np.vstack([np.zeros(shape=(T,1)), 1*np.ones(shape=(T,1))])
        
        c = np.hstack([TOU[T_on:T_off+1],-FIT[T_on:T_off+1],np.zeros((3*T,))]).reshape(-1,1) # It needs to be column matrix, though ppopt will correct it by itself.
        H = np.zeros(shape=(5*T, T))
        Q = np.zeros(shape=(5*T, 5*T))
        
        A_all = np.vstack([Aeq, A])
        b_all = np.vstack([beq, b])
        F_all = np.vstack([np.zeros(shape=(3*T, T)), F])
        
        agent.A = A
        agent.b = b
        agent.F = F
        
        agent.Aeq = Aeq
        agent.beq = beq
        agent.Feq = Feq
        
        agent.Atheta = Atheta
        agent.btheta = btheta
        
        agent.c = c
        agent.H = H
        agent.Q = Q
        
        agent.A_all = A_all
        agent.b_all = b_all
        agent.F_all = F_all
        agent.equality_indices = np.arange(Aeq.shape[0])
        
        for matrix in ['A', 'b', 'F', 'Aeq', 'beq', 'Atheta', 'btheta','c','H']:
            var = eval(matrix)
            # agent.matrix = var
            
            agent_mpcoeff[matrix][agent_id] = var
            
    return agent_mpcoeff

# def get_DOE_bd(agents_opt_info, T_on, T_off):
#     agent_bd = {} # 存储用户DOE requirement boundary
    
#     for agent,crs in agents_opt_info.items():
#         theta_lb = np.ones(shape=T_off-T_on+1)
#         theta_ub = np.ones(shape=T_off-T_on+1)
        
#         for cr_id, cr in crs.items():
#             c = cr[0]
#             E,f = cr[-2], cr[-1]
#             vertices = compute_polytope_vertices(E, f)
            
#             for vertex in vertices:
#                 theta_lb = np.minimum(vertex, theta_lb)
#                 if np.all(c==0):
#                     theta_ub = np.minimum(vertex, theta_ub)
#         agent_bd[agent,'lb'] = theta_lb
#         agent_bd[agent,'ub'] = theta_ub
        
#     agent_bd_df = pd.DataFrame.from_dict(agent_bd, orient='index')
#     agent_bd_df.index = pd.MultiIndex.from_tuples(agent_bd_df.index)
#     agent_bd_df.columns = list(range(T_on, T_off+1))
        
    # return agent_bd_df

def get_DOE_bd(agents_opt_info, T_on, T_off):
    agent_bd = {} # 存储用户DOE requirement boundary
    
    for agent,crs in agents_opt_info.items():
        # print(agent)
        theta_lb = np.ones(shape=T_off-T_on+1)
        theta_ub = np.zeros(shape=T_off-T_on+1)
        
        for cr_id, cr in crs.items():
            # print(cr_id)
            c = cr[0]
            E,f = cr[-2], cr[-1]
            
            # if np.all(c==0):
            #     pass
            try:
                vertices = compute_polytope_vertices(E, f)
                # print(agent.id, vertices)
            except Exception as e:
                print(f'*******************{agent} cannot find vertices************************')
                vertices = np.ones(shape=T_off-T_on+1)+0.01
                
            for vertex in vertices:
                theta_lb = np.minimum(vertex, theta_lb)
                
            for vertex in vertices:
                if np.any(np.abs(vertex-1)<=1e-5):
                    # print('yes')
                    # print(theta_ub)
                    # print(vertex)
                    maxi = np.where(np.abs(vertex-1)<=1e-5, 0, vertex)
                    # print(maxi)
                    theta_ub = np.maximum(maxi, theta_ub)
                    # print(theta_ub)
                    break
                else:
                    theta_ub = np.maximum(vertex, theta_ub)
                # if np.all(c==0):
                #     theta_ub = np.minimum(vertex, theta_ub)
                
                if np.any(theta_ub==1.0):
                    print('!!!!!!!!!!!!!!!!!!!!!!')
                    print('vertex',vertex)
                    print('ub:',theta_ub)
                    break
                    # print('cr_id:', cr_id, 'ub:', theta_ub)
        agent_bd[agent,'lb'] = theta_lb
        agent_bd[agent,'ub'] = theta_ub
        
    agent_bd_df = pd.DataFrame.from_dict(agent_bd, orient='index')
    agent_bd_df.index = pd.MultiIndex.from_tuples(agent_bd_df.index)
    agent_bd_df.columns = list(range(T_on, T_off+1))
        
    return agent_bd_df


def construct_feasible_region(args, prosumer_lst, inj_lb=0, inj_ub=0.1, samples=100):
    '''用于单时段，网络节点注入可行域探寻'''
    inj_samples = np.linspace(inj_lb, inj_ub, num=samples)
    prosumer_exp = {i:inj_lb for i in prosumer_lst}
    
    # test_pts = np.meshgrid(*[inj_samples]*len(prosumer_lst))
    feasible_injections = []
    num=0
    for num_i, i in enumerate(inj_samples):
        for num_j,j in enumerate(inj_samples):
            num=num+1
            if num%100==0:
                print(f'test point No. {num}, total({samples**2})')
            
            feasible_injections.append((i,j))
            prosumer_exp[prosumer_lst[0]] = i
            prosumer_exp[prosumer_lst[1]] = j
            network = grid(args, prosumer_lst = prosumer_lst, prosumers_exp = prosumer_exp)
            network.build_model()
            network.solve_model()
            network.grid_profile()
            
            if network.v.values.min()<=0.95 or network.v.values.max()>1.05:
                del feasible_injections[-1]
                
            # if num%500==0:
                # np.save(f'./feasible_pts2{num}', feasible_injections)
    return feasible_injections    
                
        
#%% Basic entities
class Args:
    def __init__(self, Lines, Nodes, Pd_df, Qd_df, v_min, v_max, v_nom, S_base,
                 TOU, FIT, lambda_0,
                 T_on, T_off, T):
        self.Lines = Lines
        self.Nodes = Nodes
        self.v_min = v_min
        self.v_max = v_max
        self.v_nom = v_nom
        self.Qd_df = Qd_df
        self.Pd_df = Pd_df
        self.S_base = S_base
        self.TOU = TOU
        self.FIT = FIT
        self.lambda_0 = lambda_0
        self.T_on, self.T_off, self.T = T_on, T_off, T
        

class agent_opt:
    '''
    INPUT:
        prosumer_id,
        SOC_0, SOC_min, SOC_max,
        P_batt_min, P_batt_max, eta,
        P_d, P_pv,
    '''
    def __init__(self, 
                 prosumer_id,
                 SOC_0, SOC_min, SOC_max,
                 P_batt_min, P_batt_max, eta,
                 P_d, P_pv,
                 # lamda, local_estimate,
                 # threshold,
                 # Psi=None, DSO_DOE=None,
                 ):
        self.id = prosumer_id
        self.SOC_0, self.SOC_min, self.SOC_max = SOC_0, SOC_min, SOC_max
        self.P_batt_min, self.P_batt_max, self.eta = P_batt_min, P_batt_max, eta
        self.P_d, self.P_pv = P_d, P_pv # 已经是求解时段的数据
        self.D = self.P_d.sum()
        
        # self.neighbour_lst = prosumer_lst[:]
        # self.neighbour_lst.remove(self.id)   
        
        # self.lamda = lamda
        # self.local_estimate = local_estimate
        # self.threshold = threshold
        # self.communication_count = 0
        # self.communication_log = []
        
        # self.Psi = Psi
        # self.DSO_DOE = DSO_DOE
        
    def build_model(self, args):
        '''
        build an agent model (gurobipy.Model) which contains all local constrs and vars

        Parameters
        ----------
        Args object.

        Returns
        -------
        None.

        '''
        self.T_on, self.T_off, self.T = args.T_on, args.T_off, args.T
        self.TOU = args.TOU
        self.FIT = args.FIT
        self.S_base = args.S_base
        
        T_on, T_off, T = self.T_on, self.T_off, self.T
        S_base = self.S_base
        TOU, FIT = self.TOU, self.FIT
        
        self.m = Model(name=f'agent_{self.id}')
        self.m.setParam('OutputFlag',0)
        
        self.P_from_grid, self.P_to_grid = {}, {}
        self.P_batt, self.SOC = {}, {}
        # felxible load
        # self.P_fl = {}
        self.P_inj={}
        # self.P_trade, self.E = {}, {}
        
        # 添加变量
        for t in range(T_on, T_off+1):
            self.P_from_grid[t] = self.m.addVar(lb=0,
                                             # ub=10,
                                             ub=GRB.INFINITY,
                                             vtype=GRB.CONTINUOUS,
                                             name=f'P_from_grid_{self.id}_{t}')
            self.P_to_grid[t] = self.m.addVar(lb=0,
                                             # ub=10,
                                             ub=GRB.INFINITY,
                                             vtype=GRB.CONTINUOUS,
                                             name=f'P_to_grid_{self.id}_{t}')
            self.P_batt[t] = self.m.addVar(
                                            # lb=-5,
                                             # ub=5,
                                             lb=-GRB.INFINITY,
                                             ub=GRB.INFINITY,
                                             vtype=GRB.CONTINUOUS,
                                             name=f'P_batt_{self.id}_{t}')
            self.SOC[t] = self.m.addVar(
                                            # lb=0,
                                              # ub=10,
                                              lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              vtype=GRB.CONTINUOUS,
                                              name=f'SOC_{self.id}_{t}') # 此处包含了SOC_0
            # self.P_trade[t] = self.m.addVar(lb=-10,
            #                                  ub=10,
            #                                  vtype=GRB.CONTINUOUS,
            #                                  name=f'P_trade_{self.id}_{t}')
            # for neighbour in self.neighbour_lst:
            #     self.E[(neighbour, t)] = self.m.addVar(lb=-10,
            #                                            ub=10,
            #                                            vtype=GRB.CONTINUOUS,
            #                                            name=f'E_{self.id}_{neighbour}_{t}')
            
            # flexible load
            # self.P_fl[t] = self.m.addVar(lb=0, 
            #                                 ub=5,
            #                                 vtype=GRB.CONTINUOUS,
            #                                 name=f'P_fl_{self.id}_{t}')
            self.P_inj[t] = self.m.addVar(
                                        # lb=-5,
                                          # ub=5,
                                          lb=-GRB.INFINITY,
                                          ub=GRB.INFINITY,
                                          vtype=GRB.CONTINUOUS,
                                          name=f'P_inj_{self.id}_{t}')
           


        # 添加约束 
        # 本地约束
        # self.m.addConstrs(((self.P_from_grid[t] - self.P_to_grid[t] == -self.P_pv[t]/S_base + self.P_batt[t] + self.P_fl[t]) for t in range(T)),
        #                  name=f'nodal_balance_{self.id}')
        self.m.addConstrs(((self.P_from_grid[t] - self.P_to_grid[t] == self.P_d[t]/S_base -self.P_pv[t]/S_base + self.P_batt[t]) for t in range(T_on, T_off+1)),
                         name=f'nodal_balance_{self.id}')
        self.m.addConstrs(((self.P_batt[t] <= self.P_batt_max/S_base) for t in range(T_on, T_off+1)),
                          name=f'P_batt_min_{self.id}')
        self.m.addConstrs(((self.P_batt[t] >= self.P_batt_min/S_base) for t in range(T_on, T_off+1)),
                          name=f'P_batt_max_{self.id}')
        # self.m.addConstr(self.SOC[T_on] == self.SOC_0/S_base,
        #                  name='SOC_0')
        # self.m.addConstrs(((self.SOC[t] == self.SOC_0/S_base + sum(self.eta*self.P_batt[tou] for tou in range(T_on, t))) for t in range(T_on+1, T_off+1)),
        #                   name=f'SOC_balance_{self.id}')  # 第二旧
        self.m.addConstr(self.SOC[T_on] == self.SOC_0/S_base + self.P_batt[T_on])
        self.m.addConstrs((self.SOC[t] == self.SOC[t-1] + self.P_batt[t]) for t in range(T_on+1, T_off+1))
        # self.m.addConstrs(((self.SOC[t] == self.SOC_0/S_base + sum(self.eta*self.P_batt[tou] for tou in range(t))) for t in range(1, T+1)),
        #                   name=f'SOC_balance_{self.id}') # 最旧：此处共T个约束
        self.m.addConstrs(((self.SOC[t] <= self.SOC_max/S_base) for t in range(T_on, T_off+1)),
                          name=f'SOC_min_{self.id}')
        self.m.addConstrs(((self.SOC[t] >= self.SOC_min/S_base) for t in range(T_on, T_off+1)),
                          name=f'SOC_max_{self.id}')
        self.m.addConstrs(((self.P_inj[t] == self.P_to_grid[t] - self.P_from_grid[t]) for t in range(T_on, T_off+1)),
                          name=f'Pinj_balance_{self.id}')
        self.m.addConstrs(((self.P_inj[t] <= 10) for t in range(T_on, T_off+1)),
                          name=f'DOE{self.id}')
        
        # self.m.addConstrs((self.P_trade[t] == sum(self.E[k] for k in self.E.keys() if k[-1] == t)) for t in range(T))
     
        # # 满足总负荷
        # self.m.addConstr(sum(self.P_fl[t] for t in range(T)) >= self.D/S_base, 
        #                   name=f'total_demand_satisfaction_{self.id}')
        

    
        
        # 松弛项
        # relaxation_term = quicksum(self.lamda[j,t] * (self.local_estimate[j,t] - self.E[(j,t)]) for j in self.neighbour_lst for t in range(T))
        # penalty_term = 0.5*rho*quicksum((self.local_estimate[j,t] - self.E[(j,t)])**2 for j in self.neighbour_lst for t in range(T))
        # 添加目标函数
        # cost_with_grid = 0
        # for t in range(T):
        #     cost_with_grid = cost_with_grid + TOU.flatten()[t] * self.P_from_grid[t] - FIT.flatten()[t] * self.P_to_grid[t]
            
        # self.obj = cost_with_grid + relaxation_term + penalty_term
        
        # 目标函数 = discomfort + purchase_energy
        # self.discomfort_cost = 0.2*sum((self.P_fl[t] - self.P_d[t]/S_base)**2 for t in range(T))
        self.purchase_cost = sum((TOU.flatten()[t] * self.P_from_grid[t] - FIT.flatten()[t] * self.P_to_grid[t]) for t in range(T_on, T_off+1))
            
        # self.obj = self.discomfort_cost + self.purchase_cost
        self.obj = self.purchase_cost
        
        # if self.Psi:
        #     self.DOE, self.P_inj = {}, {}
        #     for t in range(T):
        #         self.DOE[t] = self.m.addVar(lb=0, 
        #                                     ub=0.2,
        #                                     vtype=GRB.CONTINUOUS,
        #                                     name=f'DOE_{self.id}_{t}')
        #         # self.u[t] = self.m.addVar(vtype=GRB.BINARY,
        #         #                           name=f'u_{self.id}_{t}')
        #         self.P_inj[t] = self.m.addVar(lb=-1,
        #                                       ub=1,
        #                                       vtype=GRB.CONTINUOUS,
        #                                       name=f'P_inj_{self.id}_{t}')
                
        #     self.m.addConstrs((self.P_inj[t] == -self.P_from_grid[t] + self.P_to_grid[t] + self.P_trade[t]) for t in range(T))
        #     self.m.addConstrs((self.P_inj[t] <= self.DOE[t]) for t in range(T))
        #     # self.m.addConstrs((self.P_inj[t] >= -self.DOE[t] * (1-self.u[t])) for t in range(T))
            
        #     relaxation_term_DOE = quicksum(self.Psi[t]*(self.DSO_DOE[t] - self.DOE[t]) for t in range(T))
        #     penalty_term_DOE = 0.5*rho*quicksum((self.DSO_DOE[t] - self.DOE[t])**2 for t in range(T))
            
        #     self.obj = self.obj + relaxation_term_DOE + penalty_term_DOE
            
        
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()
        
    def solve_in_RDOE(self, DOE_df, args, appointed_t=None):
        ''' part function of test_opt in main.py'''
        DOE_given = DOE_df.loc[:, self.id]
        T_on, T_off = self.T_on, self.T_off
        TOU, FIT = self.TOU, self.FIT
        
        # rebuild the model when DOE is allocated
        self.build_model(args)
        for t in range(T_on, T_off+1):
            self.m.addConstr(self.P_inj[t]<=DOE_given[t], name=f'testDOEconstr_{t}')
            
        self.m.setParam('OutputFlag',0)
        
        if appointed_t is not None:
            try:
                self.m.optimize()
                purchase_cost = TOU.flatten()[appointed_t] * self.P_from_grid[t].x - FIT.flatten()[appointed_t] * self.P_to_grid[appointed_t].x
                exp = self.P_inj[appointed_t].x
                return exp, purchase_cost
            except Exception as e:
                # print(e)
                if self.m.Status in [3,4,5]:
                    print(f'Agent {self.id} is infeasible for this DOE.')
                    exp = 0
                    obj = 0
                    return exp, obj
                else:
                    print('Unknow error happens, not infeasible.')
        else:
            try:
                self.m.optimize()
                # if self.m.Status in [3,4,5]:
                #     print(f'Agent {self.id} is infeasible for this DOE.')
                #     exp = np.array([0 for t in range(T_on, T_off+1)])
                #     obj = 0
                #     return exp, obj
                # else:
                return self.get_exp(), self.m.getObjective().getValue()
            
            except Exception as e:
                if self.m.Status in [3,4,5]:
                    print(f'Agent {self.id} is infeasible for this DOE.')
                    # exp = np.array([0 for t in range(T_on, T_off+1)])
                    exp = DOE_given.values
                    obj = 0
                    return exp, obj
                else:
                    print('Unknow error happens, not infeasible.')
                
        
    def solve_model(self):
        self.m.setParam('OutputFlag',0)
        self.m.optimize()
        
        if self.m.Status == 3:
            print(f'{self.id} is infeasible.')
        
    def get_exp(self):
        T_on, T_off = self.T_on, self.T_off
        exp = np.array([self.P_inj[t].x for t in range(T_on, T_off+1)])
        return exp
    
    def get_vars(self):
        '''
        Get the agent's vars

        Returns
        -------
        Set:
            P_from_grid
        Set:
            P_to_grid
        Set:
            P_batt
        Set:
            SOC
        Set:
            P_inj

        '''
        return self.P_from_grid, self.P_to_grid, self.P_batt, self.SOC, self.P_inj

class agent_fl(agent_opt):
    def __init__(self,
                 prosumer_id,
                 P_pv,
                 alpha, beta,
                 d_min, d_max):
        self.id = prosumer_id
        self.P_pv = P_pv
        self.d_min, self.d_max = d_min, d_max
        self.alpha, self.beta = alpha, beta
        
    def build_model(self, args):
        self.T_on, self.T_off, self.T = args.T_on, args.T_off, args.T
        self.TOU = args.TOU
        self.FIT = args.FIT
        self.S_base = args.S_base
        
        T_on, T_off, T = self.T_on, self.T_off, self.T
        S_base = self.S_base
        TOU, FIT = self.TOU, self.FIT
        alpha = self.alpha
        beta = self.beta
        
        self.m = Model(name=f'agent_{self.id}')
        self.m.setParam('OutputFlag',0)
        
        self.P_from_grid, self.P_to_grid = {}, {}
        # felxible load
        self.P_fl = {}
        self.P_inj={}
        self.z = {}
        self.u1={}
        self.P_cur = {}
        
        # 添加变量
        for t in range(T_on, T_off+1):
            self.P_from_grid[t] = self.m.addVar(lb=0,
                                             # ub=10,
                                             ub=GRB.INFINITY,
                                             vtype=GRB.CONTINUOUS,
                                             name=f'P_from_grid_{self.id}_{t}')
            self.P_to_grid[t] = self.m.addVar(lb=0,
                                             # ub=10,
                                             ub=GRB.INFINITY,
                                             vtype=GRB.CONTINUOUS,
                                             name=f'P_to_grid_{self.id}_{t}')
            
            # flexible load
            self.P_fl[t] = self.m.addVar(lb=0, 
                                            ub=5,
                                            vtype=GRB.CONTINUOUS,
                                            name=f'P_fl_{self.id}_{t}')
            self.P_inj[t] = self.m.addVar(
                                        # lb=-5,
                                          # ub=5,
                                          lb=-GRB.INFINITY,
                                          ub=GRB.INFINITY,
                                          vtype=GRB.CONTINUOUS,
                                          name=f'P_inj_{self.id}_{t}')
            # curtailment
            self.P_cur[t] = self.m.addVar(lb=0, 
                                            ub=0.6*self.P_pv[t]/S_base,
                                            vtype=GRB.CONTINUOUS,
                                            name=f'P_cur_{self.id}_{t}')
            # 指示fl区间
            self.z[t] = self.m.addVar(vtype=GRB.BINARY, name=f'z_{self.id}_{t}')
            
            self.u1[t] = self.m.addVar(lb=0,ub=GRB.INFINITY, name=f'u1[{self.id},{t}]')
           

        # 添加约束 
        # 本地约束
        self.m.addConstrs(((self.P_from_grid[t] - self.P_to_grid[t] == self.P_fl[t] -self.P_pv[t]/S_base + self.P_cur[t]) for t in range(T_on, T_off+1)),
                         name=f'nodal_balance_{self.id}')
        self.m.addConstrs(((self.P_inj[t] == self.P_to_grid[t] - self.P_from_grid[t]) for t in range(T_on, T_off+1)),
                          name=f'Pinj_balance_{self.id}')
        self.m.addConstrs(((self.P_inj[t] <= 10) for t in range(T_on, T_off+1)),
                          name=f'DOE{self.id}')
        self.m.addConstrs((self.P_fl[t] >= self.d_min[t]/S_base for t in range(T_on, T_off+1)), 
                          name=f'min_demand_satisfaction_{self.id}')
        self.m.addConstrs((self.P_fl[t] <= self.d_max[t]/S_base for t in range(T_on, T_off+1)), 
                          name=f'max_demand_satisfaction_{self.id}')
        
        # 目标函数 = purchase_energy-utility
        for t in range(T_on, T_off+1):
            self.m.addConstr((self.z[t]==1) >> (self.P_fl[t] <= alpha/(2*beta)))
            self.m.addConstr((self.z[t]==0) >> (self.P_fl[t] >= alpha/(2*beta)))
            self.m.addConstr(self.u1[t] == alpha*self.P_fl[t] - 0.5*beta*self.P_fl[t]*self.P_fl[t])

        self.utility = sum(self.z[t]*self.u1[t] + (1-self.z[t])*0.5*alpha**2/beta for t in range(T_on, T_off+1))
        self.purchase_cost = sum((TOU.flatten()[t] * self.P_from_grid[t] - FIT.flatten()[t] * self.P_to_grid[t]) for t in range(T_on, T_off+1))
        self.curtailment_cost = sum(0.1*self.P_cur[t] for t in range(T_on, T_off+1))
            
        self.obj = self.purchase_cost - self.utility + self.curtailment_cost
        
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()
        self.m.setParam('NonConvex',2)
        
class DSO:
    def __init__(self, args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None):
        self.args = args
        self.prosumer_lst = prosumer_lst
        self.T = args.T
        self.T_on, self.T_off = args.T_on, args.T_off
        
        Lines_reformulated = args.Lines.rename(columns={'fbus':'to', 'tbus':'from', 'rateA':'s'})
        Lines_reformulated = Lines_reformulated.loc[:,~Lines_reformulated.columns.duplicated()]
        self.lines = Lines_reformulated.set_index('from', drop=False)
        self.nodes = args.Nodes
        self.v_min = args.v_min
        self.v_max = args.v_max
        
        # self.DOE_collection = DOE_collection
        # self.u_collection = u_collection
        # self.Psi = Psi
        
    def introduce_DOE(self, agents_Pmax):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        
        M = Model('DSO')
        DOE = {}
        DOE_MVar = {}

        for prosumer_node in prosumer_lst:
            for t in range(T_on, T_off+1):
                DOE[prosumer_node, t] = M.addVar(
                                                # lb=-0.01,
                                                ub=agents_Pmax.loc[prosumer_node, t],
                                                lb=0,
                                                # ub=0,
                                                
                                                # ub=0.5,
                                                name=f'DOE_[{prosumer_node,t}]')
         
        vf, vf_sum = 0,0 # 占位，set_obj有用
        self.vf = vf
        self.DOE = DOE
        self.m = M
        self.vf_sum = vf_sum
        
        return M
    def gather_bidding(self, agents_opt_info_origin, agents_Pmax):
        agents_opt_info = deepcopy(agents_opt_info_origin) # 字典可变 需要创建副本
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        
        M = Model('DSO')
        DOE = {}
        DOE_MVar = {}
        vf = {}
        z = {} # indicates theta in which cr
        y = {}

        for prosumer_node in prosumer_lst:
            for t in range(T_on, T_off+1):
                DOE[prosumer_node, t] = M.addVar(lb=0,
                                                ub=agents_Pmax.loc[prosumer_node, t],
                                                # ub=0.5,
                                                name=f'DOE_[{prosumer_node,t}]')
                
            DOE_MVar[prosumer_node] = MVar.fromlist([v for k,v in DOE.items() if k[0]==prosumer_node])


        for prosumer_node in prosumer_lst:
            # 获得该prosumer的vf下界，同时也是DOE的ub
            opt_info = agents_opt_info[prosumer_node]
            for cr_id in list(opt_info.keys()): # 字典便利中不能对字典删改，因此用list循环
                cr = opt_info[cr_id]
                if (cr[0]==0).all():
                    E,f = cr[-2], cr[-1]
                    vertices=compute_polytope_vertices(E, f)
                    vf_lb = cr[1].item()
                    # print('find!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # del opt_info[cr_id] # 删除该critical region, 该region不影响DOE分配结果，因为左端点可以由另一个cr取到       
            
            vf[prosumer_node] = M.addVar(
                                            # lb=-2,
                                            lb=vf_lb-1e-4,
                                            # lb=-0.5,
                                            ub=0.5,
                                            name=f'vf_[{prosumer_node}]')
        
        # 以下约束会使得DOE在优化过程中必须在用户上报区间内（保证DOE下发后用户可行）
        # value function / DOE preference relevant constrs list
        prosumers_relevant_constrs = []
        for agent_id in prosumer_lst:
            agent_opt_info = agents_opt_info[agent_id]
            
            if len(agent_opt_info)==0: # Only one CR
                _ = M.addConstr(vf[agent_id] == agent_opt_info[0][1].item()) # 可能有问题，时段 # 确实有问题，使用Len=0删掉if了
                prosumers_relevant_constrs.append(_)
                
            else:
                ## try offical method start
                z[agent_id] = M.addVars(agent_opt_info.keys(), vtype=GRB.BINARY, name=f'z_{agent_id}')
                for cr_id, opt_info in agent_opt_info.items():
                    vf_coeff_t = opt_info[0]
                    vf_b = opt_info[1]
                    E = opt_info[-2]
                    f = opt_info[-1]
                    
                    for row in range(E.shape[0]):
                        _ = M.addConstr((z[agent_id][cr_id]==1) >> (E[row,:]@DOE_MVar[agent_id] - f.reshape(-1)[row] <= 0)) # Etheta<=f 每行均满足
                        prosumers_relevant_constrs.append(_)
                        
                    _ = M.addConstr(z[agent_id].sum()==1) # 仅一个cr
                    prosumers_relevant_constrs.append(_)
                    _ = M.addConstr((z[agent_id][cr_id]==1) >> (vf[agent_id] == vf_coeff_t@DOE_MVar[agent_id] + vf_b.item()))
                    prosumers_relevant_constrs.append(_)
                    
                    
                    
                    
                ## try offical method end
                
                
                
                
        #         z[agent_id] = {} 
        #         y[agent_id] = {}
        #         for cr_id, opt_info in agent_opt_info.items():
        #             z[agent_id][cr_id] = {}
        #             y[agent_id][cr_id] = M.addVar(vtype=GRB.BINARY,
        #                                 name=f'y_{cr_id}')
        #             vf_coeff_t = opt_info[0]
        #             vf_b = opt_info[1]
        #             E = opt_info[-2]
        #             f = opt_info[-1]
                    
        #             z[agent_id][cr_id] = M.addVars(E.shape[0],        # var_name is cr index
        #                                   vtype=GRB.BINARY,
        #                                   name=f'z_{cr_id}')

        #             is_in_cr = E@DOE_MVar[agent_id] - f.reshape(-1) # cr polytope
        #             for row in range(E.shape[0]): # each row of polytope is satisfied or not
        #                 _ = M.addConstr((z[agent_id][cr_id][row]==1) >> (is_in_cr[row] <= 0))
        #                 prosumers_relevant_constrs.append(_)
                        
        #             _ = M.addConstr(y[agent_id][cr_id]==and_(z[agent_id][cr_id].values()), name=f'agent[{agent_id}]_cr{cr_id}')
        #             prosumers_relevant_constrs.append(_)
        #             _ = M.addConstr((y[agent_id][cr_id]==1) >> (vf[agent_id] == vf_coeff_t@DOE_MVar[agent_id] + vf_b.item())) # 选取vf
        #             prosumers_relevant_constrs.append(_)
        #             # _ = M.addConstr((y[agent_id][cr_id]==1) >> (E@DOE_MVar[agent_id] - f.reshape(-1)<=0))
        #             # prosumers_relevant_constrs.append(_)
        #         _ = M.addConstr(sum(y[agent_id][cr_id] for cr_id in range(len(agent_opt_info)))==1)
        #         prosumers_relevant_constrs.append(_)
        #             # M.addGenConstrIndicator((is_in_cr <= 0), True, (vf[agent_id] == vf_coeff_t@DOE_MVar[agent_id] + vf_b.item()))
        # #             # M.addConstr(is_in_cr <= 0)
        vf_sum = sum(vf[agent_id] for agent_id in prosumer_lst)

        
        self.vf = vf
        self.z, self.y = z, y
        self.DOE = DOE
        self.DOE_MVar = DOE_MVar
        self.m = M
        self.vf_sum = vf_sum
        self.prosumers_relevant_constrs = prosumers_relevant_constrs
        
        return M
        
    def build_model(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        lambda_0 = self.args.lambda_0
        
        self.P_0, self.Q_0 = {}, {}
        self.f_p, self.f_q = {}, {}
        self.l, self.v = {}, {}
        # self.DOE = {}
        
        # 添加网络变量+DOE
        for t in range(T_on, T_off+1):
            self.P_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'P_0_{t}')
            self.Q_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'Q_0_{t}')
            # for prosumer_node in prosumer_lst:
            #     self.DOE[prosumer_node, t] = self.m.addVar(lb=0,
            #                                                ub=0.5,
            #                                                name=f'DOE_[{prosumer_node},{t}]')
            for node in self.nodes.index:
                self.v[node, t] = self.m.addVar(lb=0, ub=2, name=f'v_{node}_{t}')
            for line in self.lines.index:
                self.l[line, t] = self.m.addVar(lb=0, 
                                                ub=10, 
                                                name=f'l_{line}_{t}')            # 此处源代码未设置lb=0
                self.f_p[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_p_{line}_{t}')
                self.f_q[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_q_{line}_{t}')
        # 添加网络约束    
        for t in range(T_on, T_off+1):
            # 电压约束
            self.m.addConstr(self.v[1,t] == 1)
            self.m.addConstrs((self.v[i,t] <= self.v_max**2) for i in self.nodes.index)
            self.m.addConstrs((self.v[i,t] >= self.v_min**2) for i in self.nodes.index)
            # 潮流约束
            self.m.addConstrs(self.f_p[i,t]**2 + self.f_q[i,t]**2 <= (self.lines['s'][i]/S_base)**2 for i in self.lines.index)
            self.m.addConstrs(((self.f_p[i,t] - self.l[i,t]*self.lines['r'][i])**2 + (self.f_q[i,t] - self.l[i,t]*self.lines['x'][i])**2 <= (self.lines['s'][i]/S_base)**2 ) for i in self.lines.index)
            self.m.addConstrs((self.v[i,t] - 2*(self.lines['r'][i]*self.f_p[i,t] + self.lines['x'][i]*self.f_q[i,t]) + self.l[i,t]*(self.lines['r'][i]**2 + self.lines['x'][i]**2) == self.v[self.lines['to'][i],t]) for i in self.lines.index)
            self.m.addConstrs((self.f_p[i,t]**2 + self.f_q[i,t]**2 <= self.l[i,t]*self.v[i,t]) for i in self.lines.index)
            # 节点约束
            for i in self.nodes.index:
                child_lines = self.lines[self.lines['to']==i]
                
                if i==1:
                    self.m.addConstr(-quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.P_0[t] == 0)
                    self.m.addConstr(-quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.Q_0[t] - self.nodes.loc[i, 'Bs']*self.v[1,t] == 0)
                elif i in prosumer_lst:
                    # 框柱DOE
                    # self.m.addConstr(self.DOE[i,t]<=self.DOE_collection[i][t])
                    # 加判断是import还是export
                    # if self.u_collection[i][t]==1:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                    # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # else:
                        # self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                        # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # M.addConstr(f_p[ancestor_lines_id,t] - quicksum(f_p[j,t]-l[j,t]*Lines['r'][j] for j in child_lines.index) - DOE[i,t] == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                else:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) + Pd_df.loc[i,t]/S_base == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t] == 0)
        
        # 设置目标
        self.cost_with_TSO = quicksum(self.P_0[t]*lambda_0[t] for t in range(T_on, T_off+1))
        
        # relaxation_term = quicksum(self.Psi[i][t]*(-self.DOE_collection[i][t] + self.DOE[i,t]) for i in prosumer_lst for t in range(T))
        # penalty_term = 0.5*rho*quicksum((-self.DOE_collection[i][t] + self.DOE[i,t])**2 for i in prosumer_lst for t in range(T))
        
        # self.obj = cost_with_TSO + relaxation_term + penalty_term
        
        # exp_sum = quicksum(self.DOE[i,t] for i in prosumer_lst for t in range(T))
        self.obj = 0 # default obj
        # self.obj = cost_with_TSO + self.vf_sum
        # self.obj = self.log_vf_sum
        # self.m.setObjective(self.obj, sense=GRB.MAXIMIZE)
        
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()
        
    def build_nonconvexmodel(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        lambda_0 = self.args.lambda_0
        
        self.P_0, self.Q_0 = {}, {}
        self.f_p, self.f_q = {}, {}
        self.l, self.v = {}, {}
        # self.DOE = {}
        
        # 添加网络变量+DOE
        for t in range(T_on, T_off+1):
            self.P_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'P_0_{t}')
            self.Q_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'Q_0_{t}')
            # for prosumer_node in prosumer_lst:
            #     self.DOE[prosumer_node, t] = self.m.addVar(lb=0,
            #                                                ub=0.5,
            #                                                name=f'DOE_[{prosumer_node},{t}]')
            for node in self.nodes.index:
                self.v[node, t] = self.m.addVar(lb=0, ub=2, name=f'v_{node}_{t}')
            for line in self.lines.index:
                self.l[line, t] = self.m.addVar(lb=0, 
                                                ub=10, 
                                                name=f'l_{line}_{t}')            # 此处源代码未设置lb=0
                self.f_p[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_p_{line}_{t}')
                self.f_q[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_q_{line}_{t}')
        # 添加网络约束    
        for t in range(T_on, T_off+1):
            # 电压约束
            self.m.addConstr(self.v[1,t] == 1)
            self.m.addConstrs((self.v[i,t] <= self.v_max**2) for i in self.nodes.index)
            self.m.addConstrs((self.v[i,t] >= self.v_min**2) for i in self.nodes.index)
            # 潮流约束
            self.m.addConstrs(self.f_p[i,t]**2 + self.f_q[i,t]**2 <= (self.lines['s'][i]/S_base)**2 for i in self.lines.index)
            self.m.addConstrs(((self.f_p[i,t] - self.l[i,t]*self.lines['r'][i])**2 + (self.f_q[i,t] - self.l[i,t]*self.lines['x'][i])**2 <= (self.lines['s'][i]/S_base)**2 ) for i in self.lines.index)
            self.m.addConstrs((self.v[i,t] - 2*(self.lines['r'][i]*self.f_p[i,t] + self.lines['x'][i]*self.f_q[i,t]) + self.l[i,t]*(self.lines['r'][i]**2 + self.lines['x'][i]**2) == self.v[self.lines['to'][i],t]) for i in self.lines.index)
            self.m.addConstrs((self.f_p[i,t]**2 + self.f_q[i,t]**2 == self.l[i,t]*self.v[i,t]) for i in self.lines.index)
            # 节点约束
            for i in self.nodes.index:
                child_lines = self.lines[self.lines['to']==i]
                
                if i==1:
                    self.m.addConstr(-quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.P_0[t] == 0)
                    self.m.addConstr(-quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.Q_0[t] - self.nodes.loc[i, 'Bs']*self.v[1,t] == 0)
                elif i in prosumer_lst:
                    # 框柱DOE
                    # self.m.addConstr(self.DOE[i,t]<=self.DOE_collection[i][t])
                    # 加判断是import还是export
                    # if self.u_collection[i][t]==1:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                    # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # else:
                        # self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                        # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # M.addConstr(f_p[ancestor_lines_id,t] - quicksum(f_p[j,t]-l[j,t]*Lines['r'][j] for j in child_lines.index) - DOE[i,t] == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                else:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) + Pd_df.loc[i,t]/S_base == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t] == 0)
        
        # 设置目标
        self.cost_with_TSO = quicksum(self.P_0[t]*lambda_0[t] for t in range(T_on, T_off+1))
        
        # relaxation_term = quicksum(self.Psi[i][t]*(-self.DOE_collection[i][t] + self.DOE[i,t]) for i in prosumer_lst for t in range(T))
        # penalty_term = 0.5*rho*quicksum((-self.DOE_collection[i][t] + self.DOE[i,t])**2 for i in prosumer_lst for t in range(T))
        
        # self.obj = cost_with_TSO + relaxation_term + penalty_term
        
        # exp_sum = quicksum(self.DOE[i,t] for i in prosumer_lst for t in range(T))
        self.obj = self.vf_sum # default obj
        # self.obj = cost_with_TSO + self.vf_sum
        # self.obj = self.log_vf_sum
        # self.m.setObjective(self.obj, sense=GRB.MAXIMIZE)
        
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()
        
    def build_LinDistFlow(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        lambda_0 = self.args.lambda_0
        
        self.P_0, self.Q_0 = {}, {}
        self.f_p, self.f_q = {}, {}
        self.l, self.v = {}, {}
        # self.DOE = {}
        
        # 添加网络变量+DOE
        for t in range(T_on, T_off+1):
            self.P_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'P_0_{t}')
            self.Q_0[t] = self.m.addVar(lb=-10,
                                        ub=10,
                                        name=f'Q_0_{t}')
            # for prosumer_node in prosumer_lst:
            #     self.DOE[prosumer_node, t] = self.m.addVar(lb=0,
            #                                                ub=0.5,
            #                                                name=f'DOE_[{prosumer_node},{t}]')
            for node in self.nodes.index:
                self.v[node, t] = self.m.addVar(lb=0, ub=2, name=f'v_{node}_{t}')
            for line in self.lines.index:
                # self.l[line, t] = self.m.addVar(lb=0, 
                #                                 ub=10, 
                #                                 name=f'l_{line}_{t}')            # 此处源代码未设置lb=0
                self.f_p[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_p_{line}_{t}')
                self.f_q[line, t] = self.m.addVar(lb=-10,
                                                  ub=10,
                                                  name=f'f_q_{line}_{t}')
        # 添加网络约束    
        for t in range(T_on, T_off+1):
            # 电压约束
            self.m.addConstr(self.v[1,t] == 1)
            self.m.addConstrs((self.v[i,t] <= self.v_max**2) for i in self.nodes.index)
            self.m.addConstrs((self.v[i,t] >= self.v_min**2) for i in self.nodes.index)
            # 潮流约束
            self.m.addConstrs(self.f_p[i,t]**2 + self.f_q[i,t]**2 <= (self.lines['s'][i]/S_base)**2 for i in self.lines.index)
            # self.m.addConstrs(((self.f_p[i,t] - self.l[i,t]*self.lines['r'][i])**2 + (self.f_q[i,t] - self.l[i,t]*self.lines['x'][i])**2 <= (self.lines['s'][i]/S_base)**2 ) for i in self.lines.index)
            self.m.addConstrs((self.v[i,t] - 2*(self.lines['r'][i]*self.f_p[i,t] + self.lines['x'][i]*self.f_q[i,t]) + 0*(self.lines['r'][i]**2 + self.lines['x'][i]**2) == self.v[self.lines['to'][i],t]) for i in self.lines.index)
            # self.m.addConstrs((self.f_p[i,t]**2 + self.f_q[i,t]**2 == self.l[i,t]*self.v[i,t]) for i in self.lines.index)
            # 节点约束
            for i in self.nodes.index:
                child_lines = self.lines[self.lines['to']==i]
                
                if i==1:
                    self.m.addConstr(-quicksum(self.f_p[j,t]-0*self.lines['r'][j] for j in child_lines.index) - self.P_0[t] == 0)
                    self.m.addConstr(-quicksum(self.f_q[j,t]-0*self.lines['x'][j] for j in child_lines.index) - self.Q_0[t] - self.nodes.loc[i, 'Bs']*self.v[1,t] == 0)
                elif i in prosumer_lst:
                    # 框柱DOE
                    # self.m.addConstr(self.DOE[i,t]<=self.DOE_collection[i][t])
                    # 加判断是import还是export
                    # if self.u_collection[i][t]==1:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-0*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                    # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # else:
                        # self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-self.l[j,t]*self.lines['r'][j] for j in child_lines.index) - self.DOE[i,t]  == 0)
                        # self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-self.l[j,t]*self.lines['x'][j] for j in child_lines.index) - self.DOE[i,t]*tan  - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                    # M.addConstr(f_p[ancestor_lines_id,t] - quicksum(f_p[j,t]-l[j,t]*Lines['r'][j] for j in child_lines.index) - DOE[i,t] == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-0*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t]== 0)
                else:
                    self.m.addConstr(self.f_p[i,t] - quicksum(self.f_p[j,t]-0*self.lines['r'][j] for j in child_lines.index) + Pd_df.loc[i,t]/S_base == 0)
                    self.m.addConstr(self.f_q[i,t] - quicksum(self.f_q[j,t]-0*self.lines['x'][j] for j in child_lines.index) + Qd_df.loc[i,t]/S_base - self.nodes.loc[i, 'Bs']*self.v[i,t] == 0)
        
        # 设置目标
        self.cost_with_TSO = quicksum(self.P_0[t]*lambda_0[t] for t in range(T_on, T_off+1))
        
        # relaxation_term = quicksum(self.Psi[i][t]*(-self.DOE_collection[i][t] + self.DOE[i,t]) for i in prosumer_lst for t in range(T))
        # penalty_term = 0.5*rho*quicksum((-self.DOE_collection[i][t] + self.DOE[i,t])**2 for i in prosumer_lst for t in range(T))
        
        # self.obj = cost_with_TSO + relaxation_term + penalty_term
        
        # exp_sum = quicksum(self.DOE[i,t] for i in prosumer_lst for t in range(T))
        self.obj = self.vf_sum # default obj
        # self.obj = cost_with_TSO + self.vf_sum
        # self.obj = self.log_vf_sum
        # self.m.setObjective(self.obj, sense=GRB.MAXIMIZE)
        
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()    
            
    def solve_model(self, remove_constr=False, **kwargs):
        # self.m.setParam('OutputFlag', 0)
        # self.m.setParam('NonConvex', 2)
        # self.m.setParam('BarHomogeneous',1)
        # self.m.Params.NumericFocus=3
        # self.m.Params.NoRelHeurTime = 10
        # self.m.Params.TimeLimit = 600
        # self.m.Params.MIPGap = 0.6
        for param, values in kwargs.items():
            self.m.setParam(param, values)
        
        if remove_constr == True:
            for constr in self.prosumers_relevant_constrs:
                self.m.remove(constr)
            
            self.m.remove(self.y)    
            self.m.remove(self.z)
            self.m.remove(self.vf)
            
        self.m.write('./RKS-NB.lp')
        self.m.optimize()
        
        if self.m.Status == 3:
            print('DSO model is infeasible.')
            # self.m.computeIIS()
            # self.m.write("DSO_model.ilp")
        
    def allocate_DOE(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        
        bc = {}
        
        for prosumer_node in prosumer_lst:
            bc_i = {}
            
            for t in range(T_on, T_off+1):
                bc_i[t] = self.DOE[prosumer_node,t].x
            
            bc[prosumer_node] = bc_i
            
        return bc
    
    def get_m(self):
        return self.m
    
    def get_obj(self):
        return self.m.getObjective().getValue()
    
    def set_obj(self, obj_type):
        T_on, T_off = self.T_on, self.T_off
        # obj lst
        vf_sum = self.vf_sum
        surplus_sum = self.vf_sum + self.cost_with_TSO
        DOE_sum = quicksum(self.DOE.values())
        DOE_sqr_sum = quicksum(map(lambda x: x**2, self.DOE.values()))
        log_vf_sum = 0
        log_surplus_sum = 0
        log_PD_sum = 0
        log_DOE_sum = 0
        
        # new vars and constrs for log obj
        if obj_type == 'log_PD_sum':
            # use for log sum
            neg_vf = {agent_id:self.m.addVar(lb=0, ub=1, name=f'neg_vf{agent_id}') for agent_id in self.prosumer_lst}
            log_vf = {agent_id:self.m.addVar(lb=-10, ub=10, name=f'log_vf{agent_id}') for agent_id in self.prosumer_lst}
            for agent_id in self.prosumer_lst:
                self.m.addConstr(neg_vf[agent_id] == -self.vf[agent_id])
                self.m.addGenConstrLog(neg_vf[agent_id], log_vf[agent_id],)
                
            log_vf_sum = sum(log_vf[agent_id] for agent_id in self.prosumer_lst)
            # 考虑log DSO surplus
            DSO_surplus = self.m.addVar(lb=0, ub=10, name=f'DSO_surplus')
            log_DSO_surplus = self.m.addVar(lb=-10, ub=10, name=f'log_DSO_surplus')
            self.m.addConstr(DSO_surplus == -self.cost_with_TSO, name=f'neg_cost_with_TSO')
            self.m.addGenConstrLog(DSO_surplus, log_DSO_surplus, name=f'log_DSO_surplus')
            
            log_PD_sum = log_vf_sum + log_DSO_surplus
            
        if obj_type == 'log_DOE_sum':
        
            # 考虑log DOE
            total_DOE = {agent_id:self.m.addVar(lb=0, ub=5.5, name=f'total_DOE[{agent_id}]') for agent_id in self.prosumer_lst}
            log_DOE = {agent_id:self.m.addVar(lb=-1000, ub=1000, name=f'log_DOE[{agent_id}]') for agent_id in self.prosumer_lst}
            
            for agent_id in self.prosumer_lst:
                self.m.addConstr(total_DOE[agent_id] == quicksum(self.DOE[agent_id, t] for t in range(T_on, T_off+1)), name=f'log_DOE[{agent_id}]')
                self.m.addGenConstrLog(total_DOE[agent_id], log_DOE[agent_id], name=f'log_DOE[{agent_id}]')              
                    
            log_DOE_sum = quicksum(log_DOE.values())
                
        if obj_type in ['log_surplus_sum','log_vf_sum']:
            # use for log sum
            neg_vf = {agent_id:self.m.addVar(lb=0, ub=1, name=f'neg_vf{agent_id}') for agent_id in self.prosumer_lst}
            log_vf = {agent_id:self.m.addVar(lb=-10, ub=10, name=f'log_vf{agent_id}') for agent_id in self.prosumer_lst}
            for agent_id in self.prosumer_lst:
                self.m.addConstr(neg_vf[agent_id] == -self.vf[agent_id])
                self.m.addGenConstrLog(neg_vf[agent_id], log_vf[agent_id],)
                
            log_vf_sum = sum(log_vf[agent_id] for agent_id in self.prosumer_lst)
            log_surplus_sum = log_vf_sum - self.cost_with_TSO
        
        
        obj_dic = {
            'vf_sum': vf_sum,
            'surplus_sum': surplus_sum,
            'DOE_sum': -DOE_sum,
            'DOE_sqr_sum': -DOE_sqr_sum,
            'log_vf_sum':-log_vf_sum,
            'log_surplus_sum':-log_surplus_sum,
            'log_PD_sum': -log_PD_sum,
            'log_DOE_sum':-log_DOE_sum,
            }
        
        if obj_type=='DOE_sqr_sum':
            self.m.setParam('NonConvex',2)
            self.m.setParam('MIPFocus',1)
            
        # if obj_type in ['DOE_sum', 'DOE_sqr_sum', 'log_DOE_sum']:
        #     self.m.remove(self.z)
        #     self.m.remove(self.y)
        
        self.obj = obj_dic[obj_type]
        self.m.setObjective(self.obj, sense=GRB.MINIMIZE)
        self.m.update()
        # self.m.write('./CIA.lp')
        
    def build_NB_solution(self, agent_bd_df, RKS=False):
        M = self.m
        prosumer_lst = self.prosumer_lst
        T_on, T_off = self.T_on, self.T_off
        
        if RKS==False:
            self.aux1, self.aux2 = {}, {}
            for i in prosumer_lst:
                for t in range(T_on, T_off+1):
                    self.aux1[i,t] = M.addVar(lb=0, ub=1, name=f'NB_aux1[{i},{t}]')
                    self.aux2[i,t] = M.addVar(lb=-100, ub=100, name=f'NB_aux2[{i},{t}]')
            
            M.addConstrs(self.DOE[i,t]>=agent_bd_df.loc[(i,'lb'),t] for i in prosumer_lst
                                                                    for t in range(T_on, T_off+1))
            M.addConstrs(self.aux1[i,t]==self.DOE[i,t]-agent_bd_df.loc[(i,'lb'),t] for i in prosumer_lst
                                                                    for t in range(T_on, T_off+1))
            self.obj = 0
            for i in prosumer_lst:
                for t in range(T_on, T_off+1):
                    M.addGenConstrLog(self.aux1[i,t], self.aux2[i,t])
                    self.obj+=self.aux2[i,t]
            
        if RKS==True:
            self.aux1, self.aux2 = {}, {}
            for i in prosumer_lst:
                for t in range(T_on, T_off+1):
                    # aux1[i,t] = M.addVar(lb=0, ub=10, name=f'NB_aux1[{i},{t}]')
                    self.aux1[i,t] = M.addVar(lb=0, ub=10, name=f'NB_aux1[{i},{t}]')
                    self.aux2[i,t] = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'NB_aux2[{i},{t}]')
            
            M.addConstrs(self.DOE[i,t]>=agent_bd_df.loc[(i,'lb'),t] for i in prosumer_lst
                                                                    for t in range(T_on, T_off+1))
            M.addConstrs(self.DOE[i,t]<=agent_bd_df.loc[(i,'ub'),t] for i in prosumer_lst
                                                                    for t in range(T_on, T_off+1))
            # M.addConstrs(self.aux1[i,t]==(self.DOE[i,t]-agent_bd_df.loc[(i,'lb'),t]) / (agent_bd_df.loc[(i,'ub'),t]-agent_bd_df.loc[(i,'lb'),t] + 1e-7) for i in prosumer_lst
            #                                                         for t in range(T_on, T_off+1))
            # M.addConstrs(self.aux1[i,t]==((self.DOE[i,t]-agent_bd_df.loc[(i,'lb'),t])/(agent_bd_df.loc[(i,'ub'),t]-agent_bd_df.loc[(i,'lb'),t]) + sum(1-(self.DOE[j,t]-agent_bd_df.loc[(j,'lb'),t])/(agent_bd_df.loc[(j,'ub'),t]-agent_bd_df.loc[(j,'lb'),t]) for j in prosumer_lst if j!=i)/(len(prosumer_lst)-1)) for i in prosumer_lst for t in range(T_on, T_off+1))
            M.addConstrs(self.aux1[i,t]==((self.DOE[i,t]-agent_bd_df.loc[(i,'lb'),t]) + 0.3*sum((agent_bd_df.loc[(j,'ub'),t] - self.DOE[j,t]) for j in prosumer_lst if j!=i)/(len(prosumer_lst)-1)) for i in prosumer_lst for t in range(T_on, T_off+1))
            
            self.obj = 0
            for i in prosumer_lst:
                for t in range(T_on, T_off+1):
                    M.addGenConstrLog(self.aux1[i,t], self.aux2[i,t])
                    M.setParam('FuncPieces',-1)
                    self.obj+=self.aux2[i,t]
            # aux = {(i,t):M.addVar(lb=-GRB.INFINITY, name=f'inner_prod({i},{t})') for i in prosumer_lst for t in range(T_on, T_off+1)}
            # for i in prosumer_lst:
            #     for t in range(T_on, T_off+1):
                    # M.addConstr(aux[i,t] == self.DOE[i,t] - agent_bd_df.loc[(i,'lb'),t] + sum((agent_bd_df.loc[(j,'ub'),t] - self.DOE[j,t]) for j in prosumer_lst if j!=i)/(len(prosumer_lst)-1))
            # aux1 = M.addVar(lb=-GRB.INFINITY)
            # aux2 = M.addVar(lb=-GRB.INFINITY)
            
            # M.addConstr(aux1 == aux[2,10]*aux[7,10])
            # M.addConstr(aux2 == aux1*aux[12,10])
            
            # self.obj = aux2
            
        M.setObjective(-self.obj)
        self.m = M
        self.m.update()
        
    def calculate_header_limit(self):
        '''计算该网络根节点最大注入功率'''
        # self.build_nonconvexmodel()
        self.build_model()
        obj = sum(-x for x in self.P_0.values())
        self.m.setObjective(obj, sense=GRB.MAXIMIZE)
        self.solve_model(remove_constr=False, OutputFlag=0, NonConvex=2)
        P_tx_sum = obj.getValue()
        P_tx = pd.Series(self.P_0).gppd.x
        print('P_tx is ', P_tx)
        self.P_tx = P_tx
        
        return P_tx
    
    def calculate_NU(self):
        P_ex = sum(-v.x for v in self.P_0.values())
        P_ex = pd.Series(self.P_0).gppd.x
        self.P_ex = P_ex
        print('P_ex is ', P_ex)
        NU = P_ex / self.P_tx
        self.NU = NU
        print('NU is ', NU)
        
        return NU
    

        
class DSO_Lin(DSO):
    def __init__(self, grid, 
                 args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None,
                 ):
        super().__init__(args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None)
        self.init_grid = grid
        
    def static_pt(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        grid = self.init_grid
        # power flow calculation
        grid.build_model()
        grid.solve_model()
        grid.grid_profile()
        
        # get expansion pt
        v = grid.v
        v.columns=grid.params.Nodes.index
        
        f_p, f_q = grid.f_p, grid.f_q
        f_p.columns, f_q.columns = grid.params.Lines.tbus, grid.params.Lines.tbus
        
        return v, f_p, f_q
    
    def coeff_calculate(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        S_base = self.args.S_base
        grid = self.init_grid
        
        v, f_p, f_q = self.static_pt()
        f_p, f_q = f_p/S_base, f_q/S_base
        
        r = pd.Series(data=0, index=f_p.columns, dtype=np.float64) # 似乎pd.Series赋值时需要先设置dtype，不一样的话不能赋值？
        x = pd.Series(data=0, index=f_p.columns, dtype=np.float64)
        
        A = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        B = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        C = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        D = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        L = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        N = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        F = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        G = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        I = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        K = pd.DataFrame(columns=f_p.columns, index=range(T_on, T_off+1), dtype=np.float64)
        
        for line in f_p.columns: 
            r[line] = grid.params.Lines.query(f'tbus=={line}')['r']
            x[line] = grid.params.Lines.query(f'tbus=={line}')['x']
            
        A, B = r/(r**2+x**2), x/(r**2+x**2)

        for line in f_p.columns:
            C[line] = r[line]*(f_p.loc[:,line]**2 + f_q.loc[:,line]**2)/v.loc[:,line]**2
            D[line] = r[line]*2*f_p.loc[:,line]/v.loc[:,line]**2
            L[line] = r[line]*2*f_q.loc[:,line]/v.loc[:,line]**2
            N[line] = r[line]*2*(f_p.loc[:,line]**2 + f_q.loc[:,line]**2)/v.loc[:,line]**3
            
            F[line] = x[line]*(f_p.loc[:,line]**2 + f_q.loc[:,line]**2)/v.loc[:,line]**2
            G[line] = x[line]*2*f_p.loc[:,line]/v.loc[:,line]**2
            I[line] = x[line]*2*f_q.loc[:,line]/v.loc[:,line]**2
            K[line] = x[line]*2*(f_p.loc[:,line]**2 + f_q.loc[:,line]**2)/v.loc[:,line]**3
            
        coeff_lst = [A,B,C,D,L,N,F,G,I,K]
        coeff_name = ['A','B','C','D','L','N','F','G','I','K']
        coeff_dic = {k:v for k,v in zip(coeff_name, coeff_lst)}
            
        return v, f_p, f_q, (r,x), coeff_dic
    
    def build_model(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        lambda_0 = self.args.lambda_0
        
        v_hat, f_p_hat, f_q_hat, (r,x), coeff_dic = self.coeff_calculate()
        A,B,C,D,L,N,F,G,I,K = [*coeff_dic.values()]
        
        M = self.m
        lines, nodes = self.lines, self.nodes
        v_min, v_max = self.v_min, self.v_max
        # add DSO network primal vars
        # P_0, Q_0 = {}, {}
        
        f_p,f_q = {}, {}
        # f_p_0, f_q_0 = {}, {} # send end, f_{ij,t} of line i
        # f_p_1, f_q_1 = {}, {} # receive end, f_{ji,t} of line i
        v, delta = {}, {}
        P_0,Q_0 = {}, {}
        
        for t in range(T_on, T_off+1):
            for node in nodes.index:
                v[node, t] = M.addVar(lb=0, ub=2, name=f'v_{node}_{t}')
                delta[node, t] = M.addVar(lb=-1, ub=1, name=f'delta_{node}_{t}')
            for line in lines.index:
                tbus, fbus = lines.loc[line,'to'], lines.loc[line,'from']
                f_p[fbus, tbus, t] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fp0[{line},{t}]')
                f_p[tbus, fbus, t] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fp1[{line},{t}]')
                f_q[fbus, tbus, t] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fq0[{line},{t}]')
                f_q[tbus, fbus, t] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fq1[{line},{t}]')
                
            P_0[t] = M.addVar(lb=-10,
                          ub=10,
                          )   
            Q_0[t] = M.addVar(lb=-10,
                          ub=10,
                          )  
            
        # add Constrs
        # primal feasibility
        for i in nodes.index:
            for t in range(T_on, T_off+1):
                p_rhs = sum(v for k,v in f_p.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1]))
                q_rhs = sum(v for k,v in f_q.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1]))
                
                if i == 1:
                    a = M.addConstr(P_0[t] == sum(v for k,v in f_p.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])),
                                    name=f'Pbalance[{i},{t}]')
                    b = M.addConstr(Q_0[t] == sum(v for k,v in f_q.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])),
                                    name=f'Qbalance[{i},{t}]')
                    M.addConstr(v[i,t]==1)
                elif i in prosumer_lst:
                    c = M.addConstr(self.DOE[i,t] == sum(v for k,v in f_p.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])))
                    d = M.addConstr(float(-Qd_df.loc[i,t]/S_base)== sum(v for k,v in f_q.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])))
                    M.addConstr(v[i,t]>=v_min,
                                name=f'vmin[{i},{t}]')
                    M.addConstr(v[i,t]<=v_max,
                                name=f'vmax[{i},{t}]')
                else:
                    M.addConstr(float(-Pd_df.loc[i,t]/S_base) ==sum(v for k,v in f_p.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])),
                                name=f'Pbalance[{i},{t}]') 
                    M.addConstr(float(-Qd_df.loc[i,t]/S_base) == sum(v for k,v in f_q.items() if (k[0]==i)and(k[-1]==t)and(k[0]!=k[1])),
                                name=f'Qbalance[{i},{t}]')
                    M.addConstr(v[i,t]>=v_min,
                                name=f'vmin[{i},{t}]')
                    M.addConstr(v[i,t]<=v_max,
                                name=f'vmax[{i},{t}]')
                    

                
        for line in lines.index:
            j, i = lines.loc[line,'to'], lines.loc[line,'from']
            for t in range(T_on, T_off+1):
                # fp0_rhs = A[line]*(v[i,t]-v[j,t]) + B[line]*(delta[i,t]-delta[j,t])
                M.addConstr(f_p[i,j,t]==A[line]*(v[i,t]-v[j,t]) + B[line]*(delta[i,t]-delta[j,t]), name=f'fp0[{line},{t}]')
                
                # fp1_rhs = -f_p[i,j,t] + C.loc[t,line] + D.loc[t,line]*(f_p[i,j,t]-f_p_hat.loc[t,line]) + L.loc[t,line]*(f_q[i,j,t]-f_q_hat.loc[t,line]) - N.loc[t,line]*(v[i,t]-v_hat.loc[t,i])
                M.addConstr(f_p[j,i,t]==-f_p[i,j,t] + C.loc[t,line] + D.loc[t,line]*(f_p[i,j,t]-f_p_hat.loc[t,line]) + L.loc[t,line]*(f_q[i,j,t]-f_q_hat.loc[t,line]) - N.loc[t,line]*(v[i,t]-v_hat.loc[t,i]), 
                            name=f'fp1[{line},{t}]')
                
                # fq0_rhs = -A[line]*(delta[i,t]-delta[j,t]) + B[line]*(v[i,t]-v[j,t]) 
                M.addConstr(f_q[i,j,t]==-A[line]*(delta[i,t]-delta[j,t]) + B[line]*(v[i,t]-v[j,t]), name=f'fq0[{line},{t}]')
                
                # fq1_rhs = -f_q[i,j,t] + F.loc[t,line] + G.loc[t,line]*(f_p[i,j,t]-f_p_hat.loc[t,line]) + I.loc[t,line]*(f_q[i,j,t]-f_q_hat.loc[t,line]) - K.loc[t,line]*(v[i,t]-v_hat.loc[t,i])
                M.addConstr(f_q[j,i,t]==-f_q[i,j,t] + F.loc[t,line] + G.loc[t,line]*(f_p[i,j,t]-f_p_hat.loc[t,line]) + I.loc[t,line]*(f_q[i,j,t]-f_q_hat.loc[t,line]) - K.loc[t,line]*(v[i,t]-v_hat.loc[t,i]),
                              name=f'fq1[{line},{t}]')
                
        M.addConstrs((delta[1,t]==0 for t in range(T_on, T_off+1)), name='delta')
        M.addConstrs((v[1,t]==1 for t in range(T_on, T_off+1)), name='voltage_ref')
        
        self.cost_with_TSO = quicksum(P_0[t]*lambda_0[t] for t in range(T_on, T_off+1))
        self.obj = 0
        
        M.setObjective(self.obj, sense=GRB.MINIMIZE)
        M.update()
        
        self.m = M
        self.f_p,self.f_q = f_p,f_q
        # f_p_0, f_q_0 = {}, {} # send end, f_{ij,t} of line i
        # f_p_1, f_q_1 = {}, {} # receive end, f_{ji,t} of line i
        self.v, self.delta = v, delta
        self.P_0,self.Q_0 = P_0,Q_0
        
class DSO_CIA(DSO):
    def __init__(self,target, grid, args, prosumer_lst, 
                 DOE_collection=None, u_collection=None, Psi=None):
        super().__init__(args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None)
        self.init_grid = grid
        self.target = target
        
    def static_pt(self):
        ''' used to get x0: 3*1 '''
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        grid = self.init_grid
        # power flow calculation
        grid.build_model()
        grid.solve_model()
        grid.grid_profile()
        
        # get expansion pt
        v = grid.v
        v.columns=grid.params.Nodes.index
        
        f_p, f_q = grid.f_p, grid.f_q
        f_p.columns, f_q.columns = grid.params.Lines.tbus, grid.params.Lines.tbus
        
        return v, f_p, f_q
    
    def coeff_calculate(self):
        S_base = self.args.S_base
        grid = self.init_grid
        target = self.target
        N = self.nodes.shape[0] - 1
        
        B = grid.net.incidence_matrix().toarray() # original incidence matrix
        B[np.nonzero(B)] = 1 # paper's incidence matrix
        
        if isinstance(target,int):
            PL = grid.net.loads_t.p.loc[target].values / S_base # 不知道为什么p_set不对
            QL = grid.net.loads_t.q.loc[target].values / S_base
        else:
            PL=0 # 占位,后续build_model不使用
            QL=0
        
        R = np.diag(self.lines.r)
        X = np.diag(self.lines.x)
        Z = R + X*(1j)
        Z2 = np.abs(Z**2) # 不确定
        # Z2 = Z**2
        
        O = np.zeros((N,1))
        I = np.identity(N)
        A = np.block([O, I])@B - I
        
        C = np.linalg.inv(I - A)
        Mp = 2*np.linalg.multi_dot([C.T, R, C])
        Mq = 2*np.linalg.multi_dot([C.T, X, C])
        
        DR = np.linalg.inv(I - A)@A@R
        DX = np.linalg.inv(I - A)@A@X
        H = np.dot(C.T, (2*(R@DR + X@DX) + Z2))   # 我的H不是对称的
        
        DX_1 = np.where(DX>0, DX, 0)
        DX_2 = np.where(DX<0 , DX, 0)
        H_1 = np.where(H>0, H, 0)
        H_2 = np.where(H<0, H, 0)
        
        # obtain operational point relevant data
        V, f_p, f_q = self.static_pt() #此处是多时段结果
        f_p, f_q = f_p/S_base, f_q/S_base
        
        if isinstance(target, int):
            x = {} # operational point: active line flow, reactive flow & volatge square
            J = {}
            He = {}
            
            for line in self.lines.index:
                # i,j = self.lines.loc[line,'to'], self.lines.loc[line,'from']
                p, q, v = f_p.loc[target, line], f_q.loc[target, line], V.loc[target, line]**2
                print(f'line:{line} active power is {p}')
                x[line] = np.array([p, q, v])
                J[line] = np.array([2*p/v,
                                   2*q/v,
                                   -(p**2+q**2)/v**2])
                He[line] = np.array([
                    [2/v, 0, -2*p/v**2],
                    [0, 2/v, -2*q/v**2],
                    [-2*p/v**2, -2*q/v**2, 2*(p**2+q**2)/v**3]
                    ])
            
            coeff_name = [
                'PL','QL','R', 'X', 'Z', 'Z2',
                'A', 'C', 'Mp', 'Mq',
                'DR', 'DX', 'H', 'DX_1', 'DX_2', 'H_1', 'H_2',
                'x', 'J', 'He'
                          ]
            coeff_var = [
                PL,QL,R, X, Z, Z2,
                A, C, Mp, Mq,
                DR, DX, H, DX_1, DX_2, H_1, H_2,
                x, J, He
                          ]
        else:
            T_on, T_off = self.T_on, self.T_off
            x = {t:{} for t in range(T_on, T_off+1)} # operational point: active line flow, reactive flow & volatge square
            J = {t:{} for t in range(T_on, T_off+1)}
            He = {t:{} for t in range(T_on, T_off+1)}
            
            for t in range(T_on, T_off+1):
                for line in self.lines.index:
                    # i,j = self.lines.loc[line,'to'], self.lines.loc[line,'from']
                    p, q, v = f_p.loc[t, line], f_q.loc[t, line], V.loc[t, line]**2
                    
                    x[t][line] = np.array([p, q, v])
                    J[t][line] = np.array([2*p/v,
                                       2*q/v,
                                       -(p**2+q**2)/v**2])
                    He[t][line] = np.array([
                        [2/v, 0, -2*p/v**2],
                        [0, 2/v, -2*q/v**2],
                        [-2*p/v**2, -2*q/v**2, 2*(p**2+q**2)/v**3]
                        ])
            
            coeff_name = [
                'PL','QL','R', 'X', 'Z', 'Z2',
                'A', 'C', 'Mp', 'Mq',
                'DR', 'DX', 'H', 'DX_1', 'DX_2', 'H_1', 'H_2',
                'x', 'J', 'He'
                          ]
            coeff_var = [
                PL,QL,R, X, Z, Z2,
                A, C, Mp, Mq,
                DR, DX, H, DX_1, DX_2, H_1, H_2,
                x, J, He
                          ]
        
        coeff_dic = {name:var for name, var in zip(coeff_name, coeff_var)}
        self.coeff_dic = coeff_dic
        
        return coeff_dic
    
    def build_model(self):
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        target = self.target
        
        coeff_dic = self.coeff_calculate()
        PL, QL ,R, X, Z, Z2, A, C, Mp, Mq, DR, DX, H, DX_1, DX_2, H_1, H_2, x, J, He = [*coeff_dic.values()]
        
        M = self.m
        lines, nodes = self.lines, self.nodes
        v_min, v_max = self.args.v_min, self.args.v_max
        N = self.nodes.shape[0] - 1
        
        if isinstance(target, int):
            # adding vars
            p,q = {}, {} # nodal injection, exclude the root node, # = N in the paper
            # pg,qg = {}, {} # don't consider the load in prosumer's nodes, so delete
            l_lb, l_ub = {}, {} # # = N in the paper, square of current magnitude
            P1, P2 = {}, {} # active power flow, P+. P-
            Q1, Q2 = {}, {} # reactive power flow, Q+, Q-
            v1, v2 = {}, {} # square of voltage, V+, V-
            l = {} # actual square of current magnitude
            P, Q = {}, {} # actual power flow
            v = {}
            delta = {} # value of delta should be a 3*1 MVar
            Psi = {}
            delta1, delta2 = {}, {} # delta+, delta-
            
            Psi_aux1, Psi_aux2, Psi_aux3, Psi_aux4 = {}, {}, {}, {}
            l_ub_aux1, l_ub_aux2, l_ub_aux3 = {}, {}, {}
            
            # root node vars and ref voltage are not needed
            
            for node in nodes.index:
                if node!=1:
                    p[node] = M.addVar(lb=-10, ub=10, name=f'p[{node}]')
                    q[node] = M.addVar(lb=-10, ub=10, name=f'q[{node}]')
                    # pg[node] = M.addVar(lb=-10, ub=10, name=f'pg[{node}]')
                    # qg[node] = M.addVar(lb=-10, ub=10, name=f'qg[{node}]')
                    v1[node] = M.addVar(lb=-10, ub=2, name=f'v1[{node}]')
                    v2[node] = M.addVar(lb=0, ub=2, name=f'v2[{node}]')
                    # v[node] = M.addVar(lb=0, ub=2, name=f'v[{node}]')
                    
            for line in lines.index:
                l_lb[line] = M.addVar(lb=-10, ub=10, name=f'l_lb[{line}]')   # 改lb了
                l_ub[line] = M.addVar(lb=0, ub=20, name=f'l_ub[{line}]')
                P1[line] = M.addVar(lb=-10, ub=10, name=f'P+[{line}]')
                P2[line] = M.addVar(lb=-10, ub=10, name=f'P-[{line}]')
                Q1[line] = M.addVar(lb=-10, ub=10, name=f'Q+[{line}]')
                Q2[line] = M.addVar(lb=-10, ub=10, name=f'Q-[{line}]')
                
                delta[line] = M.addMVar(3, lb=-10, ub=10, name=f'delta[{line}]')
                Psi[line] = M.addVar(lb=-10, ub=10, name=f'Psi[{line}]')
                # l[line] = M.addVar(lb=0, ub=10, name=f'l[{line}]')
                # P[line] = M.addVar(lb=-10, ub=10, name=f'P[{line}]')
                # Q[line] = M.addVar(lb=-10, ub=10, name=f'Q[{line}]')
                delta1[line] = M.addMVar(3, lb=-10, ub=10, name=f'delta1[{line}]')
                delta2[line] = M.addMVar(3, lb=-10, ub=10, name=f'delta2[{line}]')
                
                Psi_aux1[line] = M.addVar(lb=-10, ub=10, name=f'Psi_aux1[{line}]')
                Psi_aux2[line] = M.addVar(lb=-10, ub=10, name=f'Psi_aux2[{line}]')
                Psi_aux3[line] = M.addVar(lb=-10, ub=10, name=f'Psi_aux3[{line}]')
                Psi_aux4[line] = M.addVar(lb=-10, ub=10, name=f'Psi_aux4[{line}]')
                
                l_ub_aux1[line] = M.addVar(lb=-10, ub=10, name=f'l_ub_aux1[{line}]')
                l_ub_aux2[line] = M.addVar(lb=0, ub=10, name=f'l_ub_aux2[{line}]')
                l_ub_aux3[line] = M.addVar(lb=0, ub=10, name=f'l_ub_aux3[{line}]')
                
            # constructing MVar
            p_vec = MVar.fromlist([*p.values()])
            q_vec = MVar.fromlist([*q.values()])
            # pg_vec = MVar.fromlist([*pg.values()])
            # qg_vec = MVar.fromlist([*qg.values()])
            l_lb_vec = MVar.fromlist([*l_lb.values()])
            l_ub_vec = MVar.fromlist([*l_ub.values()])
            P1_vec = MVar.fromlist([*P1.values()])
            P2_vec = MVar.fromlist([*P2.values()])
            Q1_vec = MVar.fromlist([*Q1.values()])
            Q2_vec = MVar.fromlist([*Q2.values()])
            v1_vec = MVar.fromlist([*v1.values()])
            v2_vec = MVar.fromlist([*v2.values()])
            
            # adding constrs
            # prelim
            for line in lines.index:
                # expression of delta, delta1 & delta2
                # x_vec = [P[line], Q[line], v[line]]
                x1_vec = [P1[line], Q1[line], v1[line]]
                x2_vec = [P2[line], Q2[line], v2[line]]
                for i in range(3):
                    # M.addConstr(delta[line][i] == x_vec[i] - x[line][i], name=f'delta_expr[{line}]')
                    M.addConstr(delta1[line][i] == x1_vec[i] - x[line][i], name=f'delta1_expr[{line}]')
                    M.addConstr(delta2[line][i] == x2_vec[i] - x[line][i], name=f'delta2_expr[{line}]')
                
                # expression of psi    
                M.addConstr(Psi_aux1[line] == delta1[line].T@He[line]@delta1[line])
                M.addConstr(Psi_aux2[line] == delta1[line].T@He[line]@delta2[line])
                M.addConstr(Psi_aux3[line] == delta2[line].T@He[line]@delta1[line])
                M.addConstr(Psi_aux4[line] == delta2[line].T@He[line]@delta2[line])
                M.addConstr(Psi[line] == max_([Psi_aux1[line], Psi_aux2[line], Psi_aux3[line], Psi_aux4[line]]), name=f'Psi_expr[{line}]')
            
            # 4
            M.addConstr(P1_vec == C@p_vec - DR@l_lb_vec, name='4a')
            M.addConstr(P2_vec == C@p_vec - DR@l_ub_vec, name='4b')
            M.addConstr(Q1_vec == C@q_vec - DX_1@l_lb_vec - DX_2@l_ub_vec, name='4c')
            M.addConstr(Q2_vec == C@q_vec - DX_1@l_ub_vec - DX_2@l_lb_vec, name='4d')
            M.addConstr(v1_vec == 1*np.ones(N) + Mp@p_vec + Mq@q_vec - H_1@l_lb_vec - H_2@l_ub_vec, name='4e')
            M.addConstr(v2_vec == 1*np.ones(N) + Mp@p_vec + Mq@q_vec - H_1@l_ub_vec - H_2@l_lb_vec, name='4f')
            
            # 11,12
            for line in lines.index:
                J1 = np.where(J[line]>0, J[line], 0)
                J2 = np.where(J[line]<0, J[line], 0)
                l0 = (x[line][0]**2 + x[line][1]**2)/x[line][2]
                
                # expression of interior of abs
                M.addConstr(l_ub_aux1[line] == 2*J1.T@delta1[line] + 2*J2.T@delta2[line])
                M.addConstr(l_ub_aux2[line] == abs_(l_ub_aux1[line]))
                M.addConstr(l_ub_aux3[line] == max_(l_ub_aux2[line], Psi[line]))
                
                # 11
                # M.addConstr(l[line] <= l0 + l_ub_aux3[line], name=f'11left[{line}]') # !写错了！
                M.addConstr(l_ub[line] == l0 + l_ub_aux3[line], name=f'11right[{line}]') # 改成等于
                
                # 12
                # M.addConstr(l[line]-l0 >= J1.T@delta2[line] + J2.T@delta1[line],    # !写错了！
                #             name=f'12left[{line}]')
                M.addConstr(l_lb[line]-l0 == J1.T@delta2[line] + J2.T@delta1[line], 
                            name=f'12right[{line}]')
                
            # 13c
            for i in nodes.index:
                if i in prosumer_lst:
                    M.addConstr(p[i] == self.DOE[i, target])
                    M.addConstr(q[i] == -Qd_df.loc[i, target]/S_base)
                elif i!=1:
                    M.addConstr(p[i] == -Pd_df.loc[i, target]/S_base)
                    M.addConstr(q[i] == -Qd_df.loc[i, target]/S_base)
                else:
                    pass
    
            
            # 13d
            M.addConstr(v_min**2 <= v2_vec)
            M.addConstr(v1_vec <= v_max**2)
            
            # 13e
            M.addConstr(l_ub_vec <= 10)
            
            self.obj = self.vf_sum
            
            # calculate root node information
            P_0 = P1[2]
            # self.cost_with_TSO = P_0 * self.args.lambda_0[target]
            self.cost_with_TSO = 0 #占位
            
            M.setObjective(self.obj)
            M.update()
            
            self.P_0 = P_0
            # self.P = P
            self.m = M
            self.m.setParam('NonConvex', 2)
            
        else:
            # multi-interval CIA
            T = self.T
            T_on, T_off = self.T_on, self.T_off
            
            def gen_vars_df(phy_index, lb=-10, ub=10):
                time_index = list(range(T_on, T_off+1))
                df = pd.DataFrame()
                df.index = pd.MultiIndex.from_product([phy_index, time_index])
                df[['lb', 'ub']] = lb, ub
                return df
            
            # addding vars
            # nodal vars
            nodal_df = gen_vars_df(phy_index=nodes.index[1:])
            p, q = gppd.add_vars(M, nodal_df, name='p', lb='lb', ub='ub'),gppd.add_vars(M, nodal_df, name='q', lb='lb', ub='ub')
            v, v1, v2 = gppd.add_vars(M, nodal_df, name='v', lb=0, ub='ub'),gppd.add_vars(M, nodal_df, name='v1', lb='lb', ub='ub'),gppd.add_vars(M, nodal_df, name='v2', lb=0, ub='ub')
            
            # lines vars
            line_df = gen_vars_df(phy_index=lines.index)
            l_lb = gppd.add_vars(M, line_df, name='l_lb', lb='lb', ub='ub') 
            l_ub = gppd.add_vars(M, line_df, name='l_ub', lb=0, ub='ub')
            P1 = gppd.add_vars(M, line_df, name='P+', lb='lb', ub='ub')
            P2= gppd.add_vars(M, line_df, name='P-', lb='lb', ub='ub')
            Q1= gppd.add_vars(M, line_df, name='Q+', lb='lb', ub='ub')
            Q2= gppd.add_vars(M, line_df, name='Q-', lb='lb', ub='ub')
            
            
            Psi= gppd.add_vars(M, line_df, name='Psi', lb='lb', ub='ub')
            l= gppd.add_vars(M, line_df, name='l', lb=0, ub='ub')
            P= gppd.add_vars(M, line_df, name='P', lb='lb', ub='ub')
            Q= gppd.add_vars(M, line_df, name='Q', lb='lb', ub='ub')
            
            delta = {t:{} for t in range(T_on, T_off+1)}
            delta1 = {t:{} for t in range(T_on, T_off+1)}
            delta2 = {t:{} for t in range(T_on, T_off+1)}
            for t in range(T_on, T_off+1):
                for line in lines.index:
                    delta[t][line] = M.addMVar(3, lb=-10, ub=10, name=f'delta[{line},{t}]')
                    delta1[t][line] = M.addMVar(3, lb=-10, ub=10, name=f'delta1[{line},{t}]')
                    delta2[t][line] = M.addMVar(3, lb=-10, ub=10, name=f'delta2[{line},{t}]')
            
            Psi_aux1= gppd.add_vars(M, line_df, name='Psi_aux1', lb='lb', ub='ub')
            Psi_aux2= gppd.add_vars(M, line_df, name='Psi_aux2', lb='lb', ub='ub')
            Psi_aux3= gppd.add_vars(M, line_df, name='Psi_aux3', lb='lb', ub='ub')
            Psi_aux4= gppd.add_vars(M, line_df, name='Psi_aux4', lb='lb', ub='ub')
            
            l_ub_aux1= gppd.add_vars(M, line_df, name='l_ub_aux1', lb='lb', ub='ub')
            l_ub_aux2= gppd.add_vars(M, line_df, name='l_ub_aux2', lb=0, ub='ub')
            l_ub_aux3= gppd.add_vars(M, line_df, name='l_ub_aux3', lb=0, ub='ub')
            
            
                
            # constructing MVar
            p_vec, q_vec = {}, {}
            l_lb_vec, l_ub_vec = {}, {}
            P1_vec, P2_vec = {}, {}
            Q1_vec, Q2_vec = {}, {}
            v1_vec, v2_vec = {}, {}
            for t in range(T_on, T_off+1):
                p_vec[t] = MVar.fromlist([*p.swaplevel().loc[t]])
                q_vec[t] = MVar.fromlist([*q.swaplevel().loc[t]])
                # pg_vec = MVar.fromlist([*pg.values()])
                # qg_vec = MVar.fromlist([*qg.values()])
                l_lb_vec[t] = MVar.fromlist([*l_lb.swaplevel().loc[t]])
                l_ub_vec[t] = MVar.fromlist([*l_ub.swaplevel().loc[t]])
                P1_vec[t] = MVar.fromlist([*P1.swaplevel().loc[t]])
                P2_vec[t] = MVar.fromlist([*P2.swaplevel().loc[t]])
                Q1_vec[t] = MVar.fromlist([*Q1.swaplevel().loc[t]])
                Q2_vec[t] = MVar.fromlist([*Q2.swaplevel().loc[t]])
                v1_vec[t] = MVar.fromlist([*v1.swaplevel().loc[t]])
                v2_vec[t] = MVar.fromlist([*v2.swaplevel().loc[t]])
            
            # adding constrs
            # prelim
            for t in range(T_on, T_off+1):
                for line in lines.index:
                    # expression of delta, delta1 & delta2
                    # x_vec = [P.loc[line, t], Q.loc[line, t], v.loc[line,t]]
                    x1_vec = [P1.loc[line, t], Q1.loc[line, t], v1.loc[line, t]]
                    x2_vec = [P2.loc[line, t], Q2.loc[line, t], v2.loc[line, t]]
                    for i in range(3):
                        # M.addConstr(delta[t][line][i] == x_vec[i] - x[t][line][i], name=f'delta_expr[{line},{t}]')
                        M.addConstr(delta1[t][line][i] == x1_vec[i] - x[t][line][i], name=f'delta1_expr[{line},{t}]')
                        M.addConstr(delta2[t][line][i] == x2_vec[i] - x[t][line][i], name=f'delta2_expr[{line},{t}]')
                    
                    # expression of psi    
                    M.addConstr(Psi_aux1.loc[line, t] == delta1[t][line].T@He[t][line]@delta1[t][line])
                    M.addConstr(Psi_aux2.loc[line, t] == delta1[t][line].T@He[t][line]@delta2[t][line])
                    M.addConstr(Psi_aux3.loc[line, t] == delta2[t][line].T@He[t][line]@delta1[t][line])
                    M.addConstr(Psi_aux4.loc[line, t] == delta2[t][line].T@He[t][line]@delta2[t][line])
                    M.addConstr(Psi.loc[line, t] == max_([Psi_aux1.loc[line, t], Psi_aux2.loc[line, t], Psi_aux3.loc[line, t], Psi_aux4.loc[line, t]]), name=f'Psi_expr[{line},{t}]')
                
                # 4
                M.addConstr(P1_vec[t] == C@p_vec[t] - DR@l_lb_vec[t], name='4a[{t}]')
                M.addConstr(P2_vec[t] == C@p_vec[t] - DR@l_ub_vec[t], name='4b[{t}]')
                M.addConstr(Q1_vec[t] == C@q_vec[t] - DX_1@l_lb_vec[t] - DX_2@l_ub_vec[t], name='4c[{t}]')
                M.addConstr(Q2_vec[t] == C@q_vec[t] - DX_1@l_ub_vec[t] - DX_2@l_lb_vec[t], name='4d[{t}]')
                M.addConstr(v1_vec[t] == 1*np.ones(N) + Mp@p_vec[t] + Mq@q_vec[t] - H_1@l_lb_vec[t] - H_2@l_ub_vec[t], name='4e[{t}]')
                M.addConstr(v2_vec[t] == 1*np.ones(N) + Mp@p_vec[t] + Mq@q_vec[t] - H_1@l_ub_vec[t] - H_2@l_lb_vec[t], name='4f[{t}]')
            
                # 11,12
                for line in lines.index:
                    J1 = np.where(J[t][line]>0, J[t][line], 0)
                    J2 = np.where(J[t][line]<0, J[t][line], 0)
                    l0 = (x[t][line][0]**2 + x[t][line][1]**2)/x[t][line][2]
                    
                    # expression of interior of abs
                    M.addConstr(l_ub_aux1.loc[line, t] == 2*J1.T@delta1[t][line] + 2*J2.T@delta2[t][line])
                    M.addConstr(l_ub_aux2.loc[line, t] == abs_(l_ub_aux1.loc[line, t]))
                    M.addConstr(l_ub_aux3.loc[line, t] == max_(l_ub_aux2.loc[line, t], Psi.loc[line, t]))
                    
                    # 11
                    # M.addConstr(l_lb.loc[line, t] <= l0 + l_ub_aux3.loc[line, t], name=f'11left[{line},{t}]')
                    M.addConstr(l_ub.loc[line, t] == l0 + l_ub_aux3.loc[line, t], name=f'11right[{line},{t}]')
                    
                    # 12
                    # M.addConstr(l.loc[line, t]-l0 >= J1.T@delta1[t][line] + J2.T@delta2[t][line], 
                    #             name=f'12left[{line},{t}]')
                    M.addConstr(l_lb.loc[line, t]-l0 == J1.T@delta2[t][line] + J2.T@delta1[t][line], 
                                name=f'12right[{line},{t}]')
                
                # 13c
                for i in nodes.index:
                    if i in prosumer_lst:
                        M.addConstr(p.loc[i, t] == self.DOE[i, t])
                        M.addConstr(q.loc[i, t] == -Qd_df.loc[i, t])
                    elif i!=1:
                        M.addConstr(p.loc[i, t] == -Pd_df.loc[i, t])
                        M.addConstr(q.loc[i, t] == -Qd_df.loc[i, t])
                    else:
                        pass

            
                # 13d
                M.addConstr(v_min**2 <= v2_vec[t])
                M.addConstr(v1_vec[t] <= v_max**2)
            
                # 13e
                # M.addConstr(l_ub_vec[t] <= 10)
            
            self.obj = self.vf_sum
            
            # calculate root node information
            # P_0 = P[2] - lines.loc[2,'r']*l[2]
            # self.cost_with_TSO = P_0 * self.args.lambda_0[target]
            self.cost_with_TSO = 0
            
            M.setObjective(self.obj)
            M.update()
            
            # self.P_0 = P_0
            self.P = P
            self.m = M
            self.m.setParam('NonConvex', 2)

class MDOPF(DSO):
    def __init__(self, args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None):
        super().__init__(args, prosumer_lst, DOE_collection=None, u_collection=None, Psi=None)
        
    def build_model(self):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        M = self.m
        lambda_0 = self.args.lambda_0
        
        Pd_df = self.args.Pd_df
        Qd_df = self.args.Qd_df
        S_base = self.args.S_base
        v_min, v_max = self.args.v_min, self.args.v_max
        
        # M = Model('MDOPF')
        
        nodes_lst = self.nodes.index
        t_lst, f_lst = self.lines['to'], self.lines['from']
        tf_lst = [*zip(t_lst, f_lst)]
        ft_lst = [*zip(f_lst, t_lst)]
        lines_lst = tf_lst + ft_lst
        self.lines_lst = lines_lst
        
        # add vars
        phat = M.addVars(nodes_lst, range(T_on, T_off+1), lb=-10, name='phat') # injections
        qhat = M.addVars(nodes_lst, range(T_on, T_off+1), lb=-10, name='qhat')
        
        Phat = M.addVars(lines_lst, range(T_on, T_off+1), lb=-10, name='Phat')
        Qhat = M.addVars(lines_lst, range(T_on, T_off+1), lb=-10, name='Qhat')
        
        W = M.addVars(nodes_lst, range(T_on, T_off+1), name='W')
        V = M.addVars(nodes_lst, range(T_on, T_off+1), name='V')
        
        # add constrs
        Pflow = M.addConstrs((Phat.sum('*',j,t) + phat[j,t]==0 for j in nodes_lst for t in range(T_on, T_off+1)), name='Pflow')
        Qflow = M.addConstrs((Qhat.sum('*',j,t) + qhat[j,t]==0 for j in nodes_lst for t in range(T_on, T_off+1)), name='Qflow')
        Volaux = M.addConstrs((V[i,t] == 2 - W[i,t] for i in nodes_lst for t in range(T_on, T_off+1)), name='Volaux')
        
        for i in nodes_lst:
            if i not in prosumer_lst:
                if i != 1:
                    pinj = M.addConstrs((phat[i,t] == -Pd_df.loc[i,t]/S_base*W[i,t] + 0 for t in range(T_on, T_off+1)), name='p_injection_passive')
                    qinj = M.addConstrs((qhat[i,t] == -Qd_df.loc[i,t]/S_base*W[i,t] + 0 for t in range(T_on, T_off+1)), name='q_injection_passive')
            elif i in prosumer_lst:
                # pinj = M.addConstrs((phat[i,t] == 0 + self.DOE[i,t]*W[i,t] for t in range(T_on, T_off+1)), name='p_injection_prosumer')
                pinj = M.addConstrs((phat[i,t] == 0 + self.DOE[i,t] for t in range(T_on, T_off+1)), name='p_injection_prosumer')
                qinj = M.addConstrs((qhat[i,t] == -Qd_df.loc[i,t]/S_base*W[i,t] + 0 for i in nodes_lst if i!=1 for t in range(T_on, T_off+1)), name='q_injection_prosumer')
            

        for line in self.lines.index:
            i,j = self.lines.loc[line, 'to'], self.lines.loc[line, 'from']
            voltage = M.addConstrs((W[j,t] - W[i,t] == self.lines.loc[line, 'r']*Phat[i,j,t] + self.lines.loc[line, 'x']*Qhat[i,j,t] for t in range(T_on, T_off+1)), name=f'voltage[{line}]')
            tf_Pflow = M.addConstrs((Phat[i,j,t] == -Phat[j,i,t] for t in range(T_on, T_off+1)), name=f'tf_Pflow[{line}]')
            tf_Qflow = M.addConstrs((Qhat[i,j,t] == -Qhat[j,i,t] for t in range(T_on, T_off+1)), name=f'tf_Qflow[{line}]')
        
        vol_upper_limit = M.addConstrs((V[i,t]<=v_max for i in nodes_lst if i!=1 for t in range(T_on, T_off+1)), name='vol_upper_limit')
        vol_lower_limit = M.addConstrs((V[i,t]>=v_min for i in nodes_lst if i!=1 for t in range(T_on, T_off+1)), name='vol_lower_limit')
        ref = M.addConstrs((V[1,t] == 1 for t in range(T_on, T_off+1)), name='ref')
        
        self.cost_with_TSO = quicksum(phat[1,t]*1*lambda_0[t] for t in range(T_on, T_off+1))
        
        self.obj = self.vf_sum
        M.setObjective(self.obj, sense=GRB.MINIMIZE)
        M.update()
        
        
        self.m = M
        self.v, self.W = V, W
        self.phat, self.qhat = phat, qhat
        
    def retrieve_pinj(self):
        phat_x = pd.Series(self.phat).gppd.x
        W_x = pd.Series(self.W).gppd.x
        
        pinj = phat_x*W_x
        pinj = pinj.reset_index().pivot(index='level_0', columns='level_1')
        pinj.columns = pinj.columns.droplevel(level=0)
        pinj.columns.name='time'
        pinj.index.name='node'
        
        self.pinj = pinj
        
        return pinj
        


class grid:
    def __init__(self, args, prosumer_lst=None, prosumers_exp=None):
        self.params = args
        self.prosumer_lst = prosumer_lst
        self.prosumers_exp = prosumers_exp
        self.T_on, self.T_off, self.T = self.params.T_on, self.params.T_off, self.params.T
    
    def build_model(self):
        T_on, T_off, T = self.T_on, self.T_off, self.T
        Z_base = self.params.v_nom**2 / self.params.S_base
        S_base = self.params.S_base
        self.net = pypsa.Network()
        self.net.set_snapshots(range(T_on, T_off+1))
        
        if self.prosumer_lst is None:
            self.prosumer_lst = []
        
        # 添加节点
        for bus in self.params.Nodes.index:
            if bus == 1:
                self.net.add('Bus', f'bus_{bus}', v_nom=self.params.v_nom, v_mag_pu_set=1)
                self.net.add('Generator', 'root', bus='bus_1', control='Slack')
            
            elif bus in self.prosumer_lst:
                self.net.add('Bus', f'bus_{bus}', v_nom=self.params.v_nom)
                self.net.add('Generator', f'gen_{bus}', 
                        bus=f'bus_{bus}',
                        p_set = self.prosumers_exp[bus]*S_base,
                        # q_set = prosumers_exp[bus]*0.484*10,
                        control='PQ'
                        )
                self.net.add('Load', f'load_{bus}', 
                        bus=f'bus_{bus}',
                        p_set = 0,
                        # p_set =self.params.Pd_df.loc[bus,:].values[T_on: T_off+1],
                        q_set = self.params.Qd_df.loc[bus,:].values[T_on:T_off+1])
            else:
                self.net.add('Bus', f'bus_{bus}', v_nom=self.params.v_nom)
                self.net.add('Load', f'load_{bus}', 
                        bus=f'bus_{bus}',
                        p_set = self.params.Pd_df.loc[bus,:].values[T_on: T_off+1],
                        q_set = self.params.Qd_df.loc[bus,:].values[T_on: T_off+1],
                        )
                # if bus in prosumer_lst:
                #     net.add('Generator', f'gen_{bus}', 
                #             bus=f'bus_{bus}',
                #             p_set = prosumers_exp[bus],
                #             q_set = prosumers_exp[bus]*0.484,
                #             control='PQ'
                #             )
                
        # 添加支路
        for line in self.params.Lines.index:
            self.net.add('Line', name=f'line_{line}',
                    # bus0 = f"bus_{self.params.Lines.loc[line,'fbus']}",# 原先的tbus, fbus
                    # bus1 = f"bus_{self.params.Lines.loc[line,'tbus']}",
                    bus0 = f"bus_{self.params.Lines.loc[line,'tbus']}",
                    bus1 = f"bus_{self.params.Lines.loc[line,'fbus']}",
                    x = self.params.Lines.loc[line, 'x'] * Z_base,
                    r = self.params.Lines.loc[line, 'r'] * Z_base)
       
    def solve_model(self):
        self.net.lpf()
        self.net.pf(use_seed=True)

    def grid_profile(self):
        self.v = self.net.buses_t.v_mag_pu
        self.f_p = self.net.lines_t.p0
        self.f_q = self.net.lines_t.q0
        self.s = np.sqrt(self.f_p**2 + self.f_q**2)
        # Show_voltage(
        

        
#%% Used for gurobi
import time 

def cb_stop(model, where):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 100s
    if time.time() - model._time > 20:
        model.terminate()
#%% Market
class NB:
    def __init__(self, agent_dic, DOE_df, args):
        self.agent_dic = agent_dic
        # self.DSO_model = DSO_model
        self.DOE_df = DOE_df
        self.args = args
        
    def build_sp1(self):
        T_on, T_off = self.args.T_on, self.args.T_off
        agent_dic = self.agent_dic
        DOE_df = self.DOE_df
        TOU, FIT = self.args.TOU, self.args.FIT
        args = self.args
        S_base = args.S_base
        
        M = Model('SP1')
        
        # add agents vars
        P_from_grid = {agent_id:{} for agent_id in agent_dic.keys()}
        P_to_grid = {agent_id:{} for agent_id in agent_dic.keys()}
        P_batt = {agent_id:{} for agent_id in agent_dic.keys()}
        SOC = {agent_id:{} for agent_id in agent_dic.keys()}
        P_inj = {agent_id:{} for agent_id in agent_dic.keys()}
        DOE_ask = {agent_id:{} for agent_id in agent_dic.keys()}
        DOE = {}
        
        # store agent's obj
        purchase_cost={agent_id:None for agent_id in agent_dic.keys()}
        
        
        for agent in agent_dic.values():
            for t in range(T_on, T_off+1):
                P_from_grid[agent.id][t] = M.addVar(lb=0,
                                                  ub=10,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_from_grid_[{agent.id},{t}]')
                P_to_grid[agent.id][t] = M.addVar(lb=0,
                                                   ub=10,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_to_grid_[{agent.id},{t}]')
                P_batt[agent.id][t] = M.addVar(
                                                lb=-5,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_batt_[{agent.id},{t}]')
                SOC[agent.id][t] = M.addVar(
                                                  lb=-5,
                                                  ub=5,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'SOC_[{agent.id},{t}]') # 此处包含了SOC_0
                P_inj[agent.id][t] = M.addVar(
                                               lb=-5,
                                               ub=5,
                                              vtype=GRB.CONTINUOUS,
                                              name=f'P_inj_[{agent.id},{t}]')
                DOE_ask[agent.id][t] = M.addVar(lb=-1,
                                                  ub=1,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'A-DOE_[{agent.id},{t}]')
                DOE[agent.id, t] = M.addVar(lb=0,
                                                  ub=1,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'DOE_[{agent.id},{t}]')

        # add agent's constrs       
        for agent in agent_dic.values():
            P_d, P_pv = agent.P_d, agent.P_pv
            SOC_0, SOC_max, SOC_min = agent.SOC_0, agent.SOC_max, agent.SOC_min
            P_batt_min, P_batt_max, eta = agent.P_batt_min, agent.P_batt_max, agent.eta
            
            M.addConstrs(((P_from_grid[agent.id][t] - P_to_grid[agent.id][t] == P_d[t]/S_base -P_pv[t]/S_base + P_batt[agent.id][t]) for t in range(T_on, T_off+1)),
                              name=f'nodal_balance_[{agent.id}]')
            M.addConstrs(((P_batt[agent.id][t] <= P_batt_max/S_base) for t in range(T_on, T_off+1)),
                              name=f'P_batt_min_[{agent.id}]')
            M.addConstrs(((P_batt[agent.id][t] >= P_batt_min/S_base) for t in range(T_on, T_off+1)),
                              name=f'P_batt_max_[{agent.id}]')
            M.addConstr(SOC[agent.id][T_on] == SOC_0/S_base,
                             name='SOC_0')
            M.addConstrs(((SOC[agent.id][t] == SOC_0/S_base + sum(eta*P_batt[agent.id][tou] for tou in range(T_on, t))) for t in range(T_on+1, T_off+1)),
                              name=f'SOC_balance_[{agent.id}]') 
            M.addConstrs(((SOC[agent.id][t] <= SOC_max/S_base) for t in range(T_on, T_off+1)),
                              name=f'SOC_min_[{agent.id}]')
            M.addConstrs(((SOC[agent.id][t] >= SOC_min/S_base) for t in range(T_on, T_off+1)),
                              name=f'SOC_max_[{agent.id}]')
            M.addConstrs(((P_inj[agent.id][t] == P_to_grid[agent.id][t] - P_from_grid[agent.id][t]) for t in range(T_on, T_off+1)),
                              name=f'Pinj_balance_[{agent.id}]')
            M.addConstrs(((DOE[agent.id, t] == DOE_df.loc[t, agent.id] + DOE_ask[agent.id][t]) for t in range(T_on, T_off+1)), 
                         name=f'DOE_expr[{agent.id}]')
            M.addConstrs(((P_inj[agent.id][t] <= DOE[agent.id, t]) for t in range(T_on, T_off+1)),
                              name=f'DOE{id}')
                
            purchase_cost[agent.id] = sum((TOU.flatten()[t] * P_from_grid[agent.id][t] - FIT.flatten()[t] * P_to_grid[agent.id][t]) for t in range(T_on, T_off+1))
        
        
        # 在CIA点展开计算LinDistFlow       
        init_grid = grid(args, prosumer_lst = [*agent_dic.keys()], prosumers_exp = DOE_df)
        init_grid.build_model()
        init_grid.solve_model()
        init_grid.grid_profile()
        
        DSO_model = DSO_Lin(init_grid, args, prosumer_lst = [*agent_dic.keys()])
        DSO_model.m = M
        DSO_model.DOE = DOE
        DSO_model.vf_sum = 0 # 占位
        DSO_model.build_model() # add constrs directly in NB's model
        
        self.sp1_obj = sum(purchase_cost[agent.id] for agent in agent_dic.values())
        M.setObjective(self.sp1_obj)

        self.DOE_ask = DOE_ask
        self.DOE = DOE     
        self.purchase_cost = purchase_cost
        self.DSO_model = DSO_model
        self.sp1 = M
        
    def solve_sp1(self, **kwargs):
       self.sp1.setParam('OutputFlag', 0) 
        
       for param, values in kwargs.items():
            self.sp1.setParam(param, values)
        
       self.sp1.optimize()
        
       if self.sp1.Status == 3:
           print('SP1 is infeasible.')
           
    def get_sp1obj(self):
        prosumers_obj = {k:v.getValue() for k,v in self.purchase_cost.items()}
        return prosumers_obj
    
    def get_disaggreement(self):
        agent_dic = self.agent_dic
        DOE_df = self.DOE_df
        args = self.args
        
        prosumers_obj = {}
        for agent in agent_dic.values():
            # agent.solve_in_RDOE(DOE_df, args)
            # if agent.id != 12:
            _, prosumers_obj[agent.id] = agent.solve_in_RDOE(DOE_df, args)
        # print(prosumers_obj)
            
        return prosumers_obj
        
    
    def build_sp2(self):
        T_on, T_off = self.args.T_on, self.args.T_off
        agent_dic = self.agent_dic
        DOE_df = self.DOE_df
        TOU, FIT = self.args.TOU, self.args.FIT
        args = self.args
        S_base = args.S_base
        
        sp1_obj = self.get_sp1obj()
        disaggreement_obj = self.get_disaggreement()
        
        M = Model('SP2')
        
        # add vars
        C = {}
        aux1, aux2 = {}, {}
        for agent in agent_dic.values():
            C[agent.id] = M.addVar(lb=-10, ub=10, name=f'payment[{agent.id}]')
            aux1[agent.id] = M.addVar(lb=0, ub=10, name=f'aux1[{agent.id}]')
            aux2[agent.id] = M.addVar(lb=-100, ub=100, name=f'aux2[{agent.id}]')
            
        # add constrs
        for agent in agent_dic.values():
            M.addConstr(aux1[agent.id] == disaggreement_obj[agent.id] - sp1_obj[agent.id] - C[agent.id], name=f'in_ln[{agent.id}]')
            M.addGenConstrLog(aux1[agent.id], aux2[agent.id], name=f'ln[{agent.id}]')
            M.addConstr(aux1[agent.id] >= 0)
        M.addConstr(quicksum(C[agent.id] for agent in agent_dic.values()) == 0, name=f'payment_balance')
        
        self.sp2_obj = -quicksum(aux2[agent.id] for agent in agent_dic.values())
        M.setObjective(self.sp2_obj)
        
        self.C = C
        self.sp2 = M
        
    def solve_sp2(self, **kwargs):
        self.sp2.setParam('OutputFlag', 0) 
         
        for param, values in kwargs.items():
             self.sp2.setParam(param, values)
         
        self.sp2.optimize()
         
        if self.sp2.Status == 3:
            print('SP2 is infeasible.')
            
    def build_sp1sp(self, mp_DOE):
        T_on, T_off = self.args.T_on, self.args.T_off
        agent_dic = self.agent_dic
        TOU, FIT = self.args.TOU, self.args.FIT
        args = self.args
        S_base = args.S_base
        
        init_grid = grid(args, prosumer_lst = [*agent_dic.keys()], prosumers_exp = self.DOE_df)
        init_grid.build_model()
        init_grid.solve_model()
        init_grid.grid_profile()
        
        DSO_model = DSO_Lin(init_grid, args, prosumer_lst = [*agent_dic.keys()])
        nodes = DSO_model.nodes
        lines = DSO_model.lines
        prosumer_lst = DSO_model.prosumer_lst
        v_hat, f_p_hat, f_q_hat, (r,x), coeff_dic = DSO_model.coeff_calculate()
        A,B,C,D,L,N,F,G,I,K = [*coeff_dic.values()]
        
        M = Model('sp1sp')
        u = {}
        for t in range(T_on, T_off+1):
            for i in prosumer_lst:
                u[i,t] = M.addVar(lb=0, ub=1, name=f'u[{i},{t}]')
        # add dual vars
        alpha = {}
        lambda_p, lambda_q = {}, {}
        mu1, mu2, mu3, mu4 = {}, {}, {}, {}
        gamma1, gamma2 = {}, {}
        ksi = {}
        phi = {}
        
        for t in range(T_on, T_off+1):
            for node in nodes.index:
                lambda_p[node, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'lambda_p_[{node},{t}]')
                lambda_q[node, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'lambda_q_[{node},{t}]')

                
                if node in prosumer_lst:
                    alpha[node, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'alpha_[{node},{t}]')
                if node!=1:
                    gamma1[node, t] = M.addVar(lb=0,
                                                  ub=GRB.INFINITY,
                                                  name=f'gamma1_[{node},{t}]') # 不同于CIA的符号，这个是gamma-，表示v>V_min
                    gamma2[node, t] = M.addVar(lb=0,
                                                  ub=GRB.INFINITY,
                                                  name=f'gamma2_[{node},{t}]') # 不同于CIA的符号，这个是gamma+，表示v<V_max
                
            for line in lines.index:
                tbus, fbus = lines.loc[line,'to'], lines.loc[line,'from']
                mu1[fbus, tbus, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'mu1_[{line},{t}]')
                mu2[tbus, fbus, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'mu2_[{line},{t}]')
                mu3[fbus, tbus, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'mu3_[{line},{t}]')
                mu4[tbus, fbus, t] = M.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f'mu4_[{line},{t}]')
            ksi[t] = M.addVar(lb=-GRB.INFINITY,
                              ub=GRB.INFINITY,
                              name=f'ksi_[{t}]')
            phi[t] = M.addVar(lb=-GRB.INFINITY,
                              ub=GRB.INFINITY,
                              name=f'phi_[{t}]')
            
        # stationary pts
        for t in range(T_on, T_off+1):
            for i in nodes.index:
                
                if i in prosumer_lst:
                    # D[P_inj]
                    M.addConstr(-alpha[i,t]-lambda_p[i,t]==0, name=f'D_P_inj[{i},{t}]') 
                    # D[s_DOE]
                    # M.addConstr(1+alpha[i,t]==0, name=f'D_s_DOE[{i},{t}]')
                
                elif i == 1:
                    # D[P0] & D[Q0]
                    M.addConstr(-lambda_p[i,t]==0, name=f'D_p0[{i},{t}]') # 考虑了了根节点买卖电
                    M.addConstr(-lambda_q[i,t]==0, name=f'D_q0[{i},{t}]')
                    
                else:
                    pass
                
                # D[s_delta0]&D[s_v0]
                # M.addConstr(1+ksi[t]==0, name=f'D_s_delta0[{t}]')
                # M.addConstr(1+phi[t]==0, name=f'D_s_v0[{t}]')
                
                v_ = 0
                delta_ = 0
                for line in lines.index:
                    j, ii = int(lines.loc[line,'to']), int(lines.loc[line,'from'])
                    # if ii == i:
                    #     Ppl = 2*f_p_hat.loc[t,line]*r[line]/v_hat.loc[t,i]**2
                    #     Pql = 2*f_p_hat.loc[t,line]*x[line]/v_hat.loc[t,i]**2
                    #     Qpl = 2*f_q_hat.loc[t,line]*r[line]/v_hat.loc[t,i]**2
                    #     Qql = 2*f_q_hat.loc[t,line]*x[line]/v_hat.loc[t,i]**2
                        
                    #     # D[fp0]&D[fq0]
                    #     M.addConstr(lambda_p[i,t]-mu1[i,j,t]-mu2[j,i,t]+Ppl*mu2[j,i,t]+Pql*mu4[j,i,t]==0)
                    #     M.addConstr(lambda_q[i,t]-mu3[i,j,t]-mu4[j,i,t]+Qpl*mu2[j,i,t]+Qql*mu4[j,i,t]==0)
                    #     # # D[s_1ij]&D[s_3ij]
                    #     # M.addConstr(1+mu1[i,j,t]==0)
                    #     # M.addConstr(1+mu3[i,j,t]==0)
                    #     # D[fp1]&D[fq1]
                    #     M.addConstr(lambda_p[j,t]-mu2[j,i,t]==0)
                    #     M.addConstr(lambda_q[j,t]-mu4[j,i,t]==0)
                    #     # # D[s_1ij]&D[s_3ij]
                    #     # M.addConstr(1+mu2[j,i,t]==0)
                    #     # M.addConstr(1+mu4[j,i,t]==0)
                        
                    #     delta_ += mu1[i,j,t]*B[line] - mu3[i,j,t]*A[line]
                    #     v_ += mu1[i,j,t]*A[line] - mu2[j,i,t]*N.loc[t,line] + mu3[i,j,t]*B[line] - mu4[j,i,t]*K.loc[t,line]
                    
                    if ii==i: # 发出节点为求导目标节点
                        v_ += A[line]*mu1[i,j,t] + B[line]*mu3[i,j,t] - N.loc[t,line]*mu2[j,i,t] - K.loc[t,line]*mu4[j,i,t]
                        delta_ += B[line]*mu1[i,j,t] - A[line]*mu3[i,j,t]
                        
                    if j==i:
                        v_ += -A[line]*mu1[ii,j,t] - B[line]*mu3[ii,j,t]
                        delta_ += -B[line]*mu1[ii,j,t] + A[line]*mu3[ii,j,t]
                        
                        
                            
                # D[v]&D[delta]   
                if i!=1:
                    M.addConstr(v_ + gamma2[i,t] - gamma1[i,t] == 0, name=f'D_v[{i},{t}]')
                    M.addConstr(delta_ == 0, name=f'D_delta[{i},{t}]')
                else:
                    M.addConstr(v_ - phi[t] == 0, name=f'D_v[{i},{t}]')
                    M.addConstr(delta_ - ksi[t] == 0, name=f'D_delta[{i},{t}]')
                

                
                if i!=1:
                    # D[s_v1]&D[s_v2]
                    M.addConstr(gamma1[i,t]>=0, name=f'D_s_v1_1[{i},{t}]')
                    M.addConstr(gamma2[i,t]>=0, name=f'D_s_v1_2[{i},{t}]')
                    M.addConstr(gamma1[i,t]<=1, name=f'D_s_v2_1[{i},{t}]')
                    M.addConstr(gamma2[i,t]<=1, name=f'D_s_v2_2[{i},{t}]')
                    # # D[s_p]&D[s_q]
                    # M.addConstr(1+lambda_p[i,t]==0, name=f'D_s_p[{i},{t}]')
                    # M.addConstr(1+lambda_q[i,t]==0, name=f'D_s_q[{i},{t}]')
            for line in lines.index:
                j, i = lines.loc[line,'to'], lines.loc[line,'from']
                Ppl = 2*f_p_hat.loc[t,line]*r[line]/v_hat.loc[t,i]**2
                Pql = 2*f_p_hat.loc[t,line]*x[line]/v_hat.loc[t,i]**2
                Qpl = 2*f_q_hat.loc[t,line]*r[line]/v_hat.loc[t,i]**2
                Qql = 2*f_q_hat.loc[t,line]*x[line]/v_hat.loc[t,i]**2
                
                # D[fp0]&D[fq0]
                M.addConstr(lambda_p[i,t]-mu1[i,j,t]-mu2[j,i,t]+Ppl*mu2[j,i,t]+Pql*mu4[j,i,t]==0, name=f'D_fp0_[{i},{j},{t}]')
                M.addConstr(lambda_q[i,t]-mu3[i,j,t]-mu4[j,i,t]+Qpl*mu2[j,i,t]+Qql*mu4[j,i,t]==0)
    
                # D[fp1]&D[fq1]
                M.addConstr(lambda_p[j,t]-mu2[j,i,t]==0)
                M.addConstr(lambda_q[j,t]-mu4[j,i,t]==0)

                    
        # obj
        dpart0 = sum((alpha[i,t]*u[i,t]*mp_DOE[i,t]) for i in prosumer_lst for t in range(T_on, T_off+1))
        dpart1 = sum((lambda_p[i,t]*args.Pd_df.loc[i,t]/S_base) for i in nodes.index if i not in prosumer_lst+[1] for t in range(T_on, T_off+1)) + sum((lambda_q[i,t]*args.Qd_df.loc[i,t]/S_base) for i in nodes.index if i!=1 for t in range(T_on, T_off+1))
        dpart2 = 0
        dpart3 = 0
        for line in lines.index:
            j, i = lines.loc[line,'to'], lines.loc[line,'from']
            for t in range(T_on, T_off+1):
                dpart2 += mu2[j,i,t]*(C.loc[t,line] - D.loc[t,line]*f_p_hat.loc[t,line] - L.loc[t,line]*f_q_hat.loc[t,line] + N.loc[t,line]*v_hat.loc[t,i])
                dpart3 += mu4[j,i,t]*(F.loc[t,line] - G.loc[t,line]*f_p_hat.loc[t,line] - I.loc[t,line]*f_q_hat.loc[t,line] + K.loc[t,line]*v_hat.loc[t,i])
                
        dpart4 = sum((gamma1[i,t]*args.v_min - gamma2[i,t]*args.v_max) for i in nodes.index if i!=1 for t in range(T_on, T_off+1))
        dpart5 = sum(phi[t] for t in range(T_on, T_off+1))
        # dpart6 = sum(psi2[i,t]*3 for i in nodes.index if i!=1 for t in range(T_on, T_off+1))
        sp1sp_obj = dpart0 + dpart1 + dpart2 + dpart3 + dpart4 + dpart5 
        
        M.setObjective(-sp1sp_obj)
        M.setParam
        
        self.u = u
        self.sp1sp_obj = sp1sp_obj
        self.sp1sp = M
            
    def solve_sp1sp(self, **kwargs):
       '''返回subproblem的worst case:u的最优值以及目标值最优解''' 
       self.sp1sp.setParam('OutputFlag', 0) 
       self.sp1sp.setParam('NonConvex', 2)
        
       for param, values in kwargs.items():
            self.sp1sp.setParam(param, values)
        
       self.sp1sp.optimize()
        
       if self.sp1sp.Status == 3:
           print("SP1's'sp is infeasible.")
           
       return pd.Series(self.u).gppd.x, self.sp1sp_obj.getValue()
           
    def build_sp1mp(self,u_lst):
        T_on, T_off = self.args.T_on, self.args.T_off
        agent_dic = self.agent_dic
        DOE_df = self.DOE_df
        TOU, FIT = self.args.TOU, self.args.FIT
        args = self.args
        S_base = args.S_base
        
        M = Model('sp1mp')
        
        # Axi<=DOEi,主问题约束部分↓
        # add agents vars
        P_from_grid = {agent_id:{} for agent_id in agent_dic.keys()}
        P_to_grid = {agent_id:{} for agent_id in agent_dic.keys()}
        P_batt = {agent_id:{} for agent_id in agent_dic.keys()}
        SOC = {agent_id:{} for agent_id in agent_dic.keys()}
        P_inj = {agent_id:{} for agent_id in agent_dic.keys()}
        DOE_ask = {agent_id:{} for agent_id in agent_dic.keys()}
        DOE = {}
        
        # store agent's obj
        purchase_cost={agent_id:None for agent_id in agent_dic.keys()}
        
        
        for agent in agent_dic.values():
            for t in range(T_on, T_off+1):
                P_from_grid[agent.id][t] = M.addVar(lb=0,
                                                  ub=10,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_from_grid_[{agent.id},{t}]')
                P_to_grid[agent.id][t] = M.addVar(lb=0,
                                                   ub=10,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_to_grid_[{agent.id},{t}]')
                P_batt[agent.id][t] = M.addVar(
                                                lb=-5,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f'P_batt_[{agent.id},{t}]')
                SOC[agent.id][t] = M.addVar(
                                                  lb=-5,
                                                  ub=5,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'SOC_[{agent.id},{t}]') # 此处包含了SOC_0
                P_inj[agent.id][t] = M.addVar(
                                               lb=-5,
                                               ub=5,
                                              vtype=GRB.CONTINUOUS,
                                              name=f'P_inj_[{agent.id},{t}]')
                DOE_ask[agent.id][t] = M.addVar(lb=-1,
                                                  ub=1,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'A-DOE_[{agent.id},{t}]')
                DOE[agent.id, t] = M.addVar(lb=0,
                                                  ub=1,
                                                  vtype=GRB.CONTINUOUS,
                                                  name=f'DOE_[{agent.id},{t}]')
        # add agent's constrs       
        for agent in agent_dic.values():
            P_d, P_pv = agent.P_d, agent.P_pv
            SOC_0, SOC_max, SOC_min = agent.SOC_0, agent.SOC_max, agent.SOC_min
            P_batt_min, P_batt_max, eta = agent.P_batt_min, agent.P_batt_max, agent.eta
            
            M.addConstrs(((P_from_grid[agent.id][t] - P_to_grid[agent.id][t] == P_d[t]/S_base -P_pv[t]/S_base + P_batt[agent.id][t]) for t in range(T_on, T_off+1)),
                              name=f'nodal_balance_[{agent.id}]')
            M.addConstrs(((P_batt[agent.id][t] <= P_batt_max/S_base) for t in range(T_on, T_off+1)),
                              name=f'P_batt_min_[{agent.id}]')
            M.addConstrs(((P_batt[agent.id][t] >= P_batt_min/S_base) for t in range(T_on, T_off+1)),
                              name=f'P_batt_max_[{agent.id}]')
            M.addConstr(SOC[agent.id][T_on] == SOC_0/S_base,
                             name='SOC_0')
            M.addConstrs(((SOC[agent.id][t] == SOC_0/S_base + sum(eta*P_batt[agent.id][tou] for tou in range(T_on, t))) for t in range(T_on+1, T_off+1)),
                              name=f'SOC_balance_[{agent.id}]') 
            M.addConstrs(((SOC[agent.id][t] <= SOC_max/S_base) for t in range(T_on, T_off+1)),
                              name=f'SOC_min_[{agent.id}]')
            M.addConstrs(((SOC[agent.id][t] >= SOC_min/S_base) for t in range(T_on, T_off+1)),
                              name=f'SOC_max_[{agent.id}]')
            M.addConstrs(((P_inj[agent.id][t] == P_to_grid[agent.id][t] - P_from_grid[agent.id][t]) for t in range(T_on, T_off+1)),
                              name=f'Pinj_balance_[{agent.id}]')
            M.addConstrs(((DOE[agent.id, t] == DOE_df.loc[t, agent.id] + DOE_ask[agent.id][t]) for t in range(T_on, T_off+1)), 
                         name=f'DOE_expr[{agent.id}]')
            M.addConstrs(((P_inj[agent.id][t] <= DOE[agent.id, t]) for t in range(T_on, T_off+1)),
                              name=f'DOE{id}')
                
            purchase_cost[agent.id] = sum((TOU.flatten()[t] * P_from_grid[agent.id][t] - FIT.flatten()[t] * P_to_grid[agent.id][t]) for t in range(T_on, T_off+1))
        
        # 含y,子问题约束部分
        # 在CIA点展开计算LinDistFlow       
        init_grid = grid(args, prosumer_lst = [*agent_dic.keys()], prosumers_exp = DOE_df)
        init_grid.build_model()
        init_grid.solve_model()
        init_grid.grid_profile()
        
        DSO_model = DSO_Lin4CCG(init_grid, args, prosumer_lst = [*agent_dic.keys()])
        
        if u_lst :
            for n,u in enumerate(u_lst):
            # 通过DSO_model往M中加入网络约束
                DSO_model.m = M
                # add constr: p_inj = u*DOE, 此处的p_inj是节点可能的注入功率，上部分的P_inj是用户侧在决策自己DOE时使用的注入
                p_inj = {}
                for i in agent_dic.keys():
                    for t in range(T_on, T_off+1):
                        p_inj[i,t,n] = M.addVar(lb=-GRB.INFINITY, name='p_inj_uncertain[{i},{t},{n}]')
                        M.addConstr(p_inj[i,t,n]==u[i,t]*DOE[i,t], name='uncertain_inj[{i},{t},{n}]') # uncertain injection
                DSO_model.DOE = p_inj
                DSO_model.build_model(n)
        else:
            pass

        
        self.sp1mp_obj = sum(purchase_cost[agent.id] for agent in agent_dic.values())
        M.setObjective(self.sp1mp_obj)

        self.DOE_ask = DOE_ask
        self.DOE = DOE     
        self.purchase_cost = purchase_cost
        self.sp1mp = M
        
    def solve_sp1mp(self, **kwargs):
        '''返回master problem在本次迭代的最优DOE'''
        self.sp1mp.setParam('OutputFlag', 0) 
        
        for param, values in kwargs.items():
             self.sp1mp.setParam(param, values)
         
        self.sp1mp.optimize()
         
        if self.sp1mp.Status == 3:
            print("SP1's'mp is infeasible.")
           
        return pd.Series(self.DOE).gppd.x
            
    def build_sp1CCG(self):
        N = 10
        n = 0 # 迭代次数
        u_lst = [] # 存储worst scenario,每个元素是一个u的multiindex series
        sp_obj_lst = [] # 存储子问题目标函数最优值
        
        for i in range(N):
            print('----------------------------------------')
            print(f'CCG round {i}.')
            self.build_sp1mp(u_lst)
            DOE_opt = self.solve_sp1mp()
            
            self.build_sp1sp(DOE_opt)
            u_opt, sp_obj_opt = self.solve_sp1sp()
            
            u_lst.append(u_opt)
            sp_obj_lst.append(sp_obj_opt)
            
            if sp_obj_opt<=1e-3:
                print(f'CCG terminate in No.{i}.')
            
                self.u_lst = u_lst
                DOE_df = DOE_opt.reset_index().pivot(index='level_0', columns='level_1')
                DOE_df.columns = DOE_df.columns.droplevel(level=0)
                DOE_df.columns.name='time'
                DOE_df.index.name='agent'
                
                self.DOE_opt = DOE_df.T
                
                break
            
        
        
        
        
        
class DSO_Lin4CCG(DSO_Lin):
    def build_model(self, n):
        T = self.T
        T_on, T_off = self.T_on, self.T_off
        prosumer_lst = self.prosumer_lst
        S_base = self.args.S_base
        Pd_df, Qd_df = self.args.Pd_df, self.args.Qd_df
        lambda_0 = self.args.lambda_0
        
        v_hat, f_p_hat, f_q_hat, (r,x), coeff_dic = self.coeff_calculate()
        A,B,C,D,L,N,F,G,I,K = [*coeff_dic.values()]
        
        M = self.m
        lines, nodes = self.lines, self.nodes
        v_min, v_max = self.v_min, self.v_max
        # add DSO network primal vars
        # P_0, Q_0 = {}, {}
        
        f_p,f_q = {}, {}
        # f_p_0, f_q_0 = {}, {} # send end, f_{ij,t} of line i
        # f_p_1, f_q_1 = {}, {} # receive end, f_{ji,t} of line i
        v, delta = {}, {}
        P_0,Q_0 = {}, {}
        
        for t in range(T_on, T_off+1):
            for node in nodes.index:
                v[node, t, n] = M.addVar(lb=0, ub=2, name=f'v_{node}_{t},{n}_{n}')
                delta[node, t, n] = M.addVar(lb=-1, ub=1, name=f'delta_{node}_{t},{n}_{n}')
            for line in lines.index:
                tbus, fbus = lines.loc[line,'to'], lines.loc[line,'from']
                f_p[fbus, tbus, t,n] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fp0[{line},{t},{n}]')
                f_p[tbus, fbus, t,n] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fp1[{line},{t},{n}]')
                f_q[fbus, tbus, t,n] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fq0[{line},{t},{n}]')
                f_q[tbus, fbus, t,n] = M.addVar(lb=-15,
                                        ub=15,
                                        name=f'fq1[{line},{t},{n}]')
                
            P_0[t,n] = M.addVar(lb=-10,
                          ub=10,
                          )   
            Q_0[t,n] = M.addVar(lb=-10,
                          ub=10,
                          )  
            
        # add Constrs
        # primal feasibility
        for i in nodes.index:
            for t in range(T_on, T_off+1):
                p_rhs = sum(v for k,v in f_p.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1]))
                q_rhs = sum(v for k,v in f_q.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1]))
                
                if i == 1:
                    a = M.addConstr(P_0[t,n] == sum(v for k,v in f_p.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])),
                                    name=f'Pbalance[{i},{t},{n}]')
                    b = M.addConstr(Q_0[t,n] == sum(v for k,v in f_q.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])),
                                    name=f'Qbalance[{i},{t},{n}]')
                    # M.addConstr(v[i,t,n]==1)
                elif i in prosumer_lst:
                    c = M.addConstr(self.DOE[i,t,n] == sum(v for k,v in f_p.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])))
                    d = M.addConstr(-Qd_df.loc[i,t]/S_base== sum(v for k,v in f_q.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])))
                    M.addConstr(v[i,t,n]>=v_min,
                                name=f'vmin[{i},{t},{n}]')
                    M.addConstr(v[i,t,n]<=v_max,
                                name=f'vmax[{i},{t},{n}]')
                else:
                    M.addConstr(-Pd_df.loc[i,t]/S_base ==sum(v for k,v in f_p.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])),
                                name=f'Pbalance[{i},{t},{n}]') 
                    M.addConstr(-Qd_df.loc[i,t]/S_base == sum(v for k,v in f_q.items() if (k[0]==i)and(k[2]==t)and(k[0]!=k[1])),
                                name=f'Qbalance[{i},{t},{n}]')
                    M.addConstr(v[i,t,n]>=v_min,
                                name=f'vmin[{i},{t},{n}]')
                    M.addConstr(v[i,t,n]<=v_max,
                                name=f'vmax[{i},{t},{n}]')
                    

                
        for line in lines.index:
            j, i = lines.loc[line,'to'], lines.loc[line,'from']
            for t in range(T_on, T_off+1):
                # fp0_rhs = A[line]*(v[i,t,n]-v[j,t,n]) + B[line]*(delta[i,t,n]-delta[j,t,n])
                M.addConstr(f_p[i,j,t,n]==A[line]*(v[i,t,n]-v[j,t,n]) + B[line]*(delta[i,t,n]-delta[j,t,n]), name=f'fp0[{line},{t},{n}]')
                
                # fp1_rhs = -f_p[i,j,t,n] + C.loc[t,line] + D.loc[t,line]*(f_p[i,j,t,n]-f_p_hat.loc[t,line]) + L.loc[t,line]*(f_q[i,j,t,n]-f_q_hat.loc[t,line]) - N.loc[t,line]*(v[i,t,n]-v_hat.loc[t,i])
                M.addConstr(f_p[j,i,t,n]==-f_p[i,j,t,n] + C.loc[t,line] + D.loc[t,line]*(f_p[i,j,t,n]-f_p_hat.loc[t,line]) + L.loc[t,line]*(f_q[i,j,t,n]-f_q_hat.loc[t,line]) - N.loc[t,line]*(v[i,t,n]-v_hat.loc[t,i]), 
                            name=f'fp1[{line},{t},{n}]')
                
                # fq0_rhs = -A[line]*(delta[i,t,n]-delta[j,t,n]) + B[line]*(v[i,t,n]-v[j,t,n]) 
                M.addConstr(f_q[i,j,t,n]==-A[line]*(delta[i,t,n]-delta[j,t,n]) + B[line]*(v[i,t,n]-v[j,t,n]), name=f'fq0[{line},{t},{n}]')
                
                # fq1_rhs = -f_q[i,j,t,n] + F.loc[t,line] + G.loc[t,line]*(f_p[i,j,t,n]-f_p_hat.loc[t,line]) + I.loc[t,line]*(f_q[i,j,t,n]-f_q_hat.loc[t,line]) - K.loc[t,line]*(v[i,t,n]-v_hat.loc[t,i])
                M.addConstr(f_q[j,i,t,n]==-f_q[i,j,t,n] + F.loc[t,line] + G.loc[t,line]*(f_p[i,j,t,n]-f_p_hat.loc[t,line]) + I.loc[t,line]*(f_q[i,j,t,n]-f_q_hat.loc[t,line]) - K.loc[t,line]*(v[i,t,n]-v_hat.loc[t,i]),
                             name=f'fq1[{line},{t},{n}]')
                
        M.addConstrs((delta[1,t,n]==0 for t in range(T_on, T_off+1)), name='delta[{t},{n}]')
        M.addConstrs((v[1,t,n]==1 for t in range(T_on, T_off+1)), name='voltage_ref[{t},{n}]')
        

        