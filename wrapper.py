import os
import shutil
import datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pyqubo
from pyqubo import Array, Placeholder, solve_qubo, Constraint, Sum


class Opt():
    def __init__(self,df,department_names,term_names,region_dict,path):
        self.df,self.department_names,self.term_names,self.region_dict=df,department_names,term_names,region_dict

        self.set_Situation(df,department_names,term_names,region_dict)
        self.set_Solver()

        self.challenge()
        self.save_result(target="score",folder_path=path)

    def set_Situation(self,df,department_names,term_names,region_dict,show=True):
        self.S = Situation(df,department_names,term_names,region_dict)
        self.S.using_mean_capacity()
        if show:
            print("capacity is estimated below")
            for key,value in self.S.expectation_capacity_dict.items():
                print(key)
                print(value,end="\n\n")

    def set_Solver(self):
        self.SL = Solver(self.S)

    def challenge(self,N=20,show=True):
        MET_max = 100
        i=0
        score_dict = {"A":10000,"B":10000}
        MET_dict = {"A":10000,"B":10000}
        min_score_PP_dict = {}
        min_MET_PP_dict = {}
        if show:
            print("start trial")
        while i < N:
            i += 1
            #solve
            solution = self.SL.solve_all()
            #postprocess
            temp_PP_dict = {group_name:PostProcess(solution[group_name],self.df,self.department_names,self.term_names,self.S.group_num_dict[group_name]) for group_name in self.S.group_names}
            for key,value in temp_PP_dict.items():
                value.main(show=False)
            np_X_dict = {key:value.np_X for key,value in temp_PP_dict.items()}
            temp_score_dict = {key:value.score for key,value in temp_PP_dict.items()}
            temp_MET_dict = {key:value.MET_max for key,value in temp_PP_dict.items()}
            for group_name in self.S.group_names:
                if score_dict[group_name] > temp_score_dict[group_name]:
                    score_dict[group_name] = temp_score_dict[group_name]
                    min_score_PP_dict[group_name] = temp_PP_dict[group_name]
                if MET_dict[group_name] > temp_MET_dict[group_name]:
                    MET_dict[group_name] > temp_MET_dict[group_name]
                    min_MET_PP_dict[group_name] = temp_PP_dict[group_name]
            if show:
                print(i,temp_score_dict,temp_MET_dict)
        
        self.min_score_PP_dict,self.min_MET_PP_dict=min_score_PP_dict,min_MET_PP_dict

    def look_df(self,df):
        cm = sns.light_palette("green", as_cmap=True)
        return df.style.background_gradient(cmap=cm)

    def save_result(self,target = "score",folder_path=None):
        if target == "score":
            target_PP_dict = self.min_score_PP_dict
        elif target == "MET":
            target_PP_dict = self.min_MET_PP_dict
        else:
            raise Exception

        if folder_path is None:
            dt = datetime.datetime.now()
            folder_path = f"{dt.date()}-{dt.hour}-{dt.minute}"
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
            
        self.style_dict = {}
        self.program_dict = {}
        for group in self.S.group_names:
            names = self.df[self.df.group==group].name
            df_count = pd.DataFrame(target_PP_dict[group].np_MET_bi,index=names,columns=names)
            self.style_dict[group]=self.look_df(df_count)
            self.style_dict[group].to_excel(f"{folder_path}/MET_{group}.xlsx")
            self.program_dict[group] = target_PP_dict[group].df_program
            self.program_dict[group].applymap(lambda x : str(x).replace(","," / ")).to_csv(f"{folder_path}/program_{group}.csv")

        for i,group in enumerate(self.S.group_names):
            if i==0:
              program_all = self.program_dict[group].copy(deep=True)
            else:
              program_all = program_all + self.program_dict[group].copy(deep=True)
        program_all.applymap(lambda x : str(x).replace(","," / ")).to_csv(f"{folder_path}/program_all.csv")
        

class Situation():
  def __init__(self,df,department_names,term_names,
               region_dict,zero_member_constraint_strength = 2):
    
    self.df = df
    self.group_names = self.df.group.unique()
    self.group_num_dict = self.df.group.value_counts().to_dict()
    self.department_names = department_names
    self.department_num = len(self.department_names)
    self.term_names = term_names
    self.term_num = len(term_names)
    self.region_dict = region_dict
    self.zero_member_constraint_strength = zero_member_constraint_strength

  def using_mean_capacity(self):
    self.expectation_capacity_dict = {}
    for group_name in self.group_names:
      member_num = self.group_num_dict[group_name]
      term_active_num = np.einsum("dt->t",self.region_dict[group_name])
      if (term_active_num == 0).any():
        raise Exception("there should be no deactivate term")
      term_mean = member_num / term_active_num
      self.expectation_capacity_dict[group_name] = np.einsum("dt,t->dt",self.region_dict[group_name],term_mean)
    self.modify_zero()    

  def set_capacity(self,expectation_capacity_dict):
    # use this function, if you dont want to use mean
    self.expectation_capacity_dict = expectation_capacity_dict
    self.modify_zero()

  def modify_zero(self):
    # to empower 0 constraint, conver 0 -> -strength
    for key,value in self.expectation_capacity_dict.items():
      self.expectation_capacity_dict[key] = value - self.zero_member_constraint_strength * (value == 0)

class Utils():
  def decode_(self,sol,shape):
    np_X = np.zeros(shape,dtype=int)
    for d in range(shape[0]):
      for t in range(shape[1]):
        for m in range(shape[2]):
          np_X[d,t,m] = sol[f"X[{d}][{t}][{m}]"]
    return np_X

  def decode(self,sol):
    dmax = 0
    tmax = 0
    mmax = 0
    for key in sol.keys():
      temp = key.split("[")
      temp_d = int(temp[1][:-1])
      temp_t = int(temp[2][:-1])
      temp_m = int(temp[3][:-1])
      if dmax < temp_d:
        dmax = temp_d
      if tmax < temp_t:
        tmax = temp_t
      if mmax < temp_m:
        mmax = temp_m
    shape = (dmax+1,tmax+1,mmax+1)
    return self.decode_(sol,shape)

  def TM_test(self,np_X):
    # TM_Once test
    return (np.einsum("dtm->tm",np_X) == 1).all()

  def DM_test(self,np_X):
    # DM_Once test
    return (np.einsum("dtm->dm",np_X) == 1).all()

  def MC_test(self,np_X,region):
    # member_capacity test
    dt=np.einsum("dtm->dt",np_X) > 0 
    return (dt-region == 0).all()

  def count_MET(self,np_X,bi=False):
    member_num = np_X.shape[2]
    np_MET = np.zeros((member_num,member_num),dtype=int)
    if bi:
      for m_i in range(member_num):
        for m_j in range(member_num):
          v_i = np_X[:,:,m_i]
          v_j = np_X[:,:,m_j]
          np_MET[m_i,m_j] = np.sum(v_i*v_j)
      for m in range(member_num):
        np_MET[m,m] = 2
    else:
      for m_i in range(member_num):
        for m_j in range(m_i):
          v_i = np_X[:,:,m_i]
          v_j = np_X[:,:,m_j]
          np_MET[m_i,m_j] = np.sum(v_i*v_j)
    return np_MET

  def l2_score(self,np_X):
    np_MET = self.count_MET(np_X)
    return np.sum(np_MET**2)


class Solver(Utils):
  def __init__(self,situation):
    self.situation = situation

    self.df = situation.df

    self.department_num = situation.department_num
    self.term_num = situation.term_num

    self.group_names = situation.group_names
    self.group_num_dict = situation.group_num_dict
    self.region_dict = situation.region_dict
    self.expectation_capacity_dict = situation.expectation_capacity_dict

    self.build_all_solver()

  def build_all_solver(self):
    self.model_dict = {group_name:self.build_solver(self.group_num_dict[group_name],
                                                    self.expectation_capacity_dict[group_name]) for group_name in self.group_names}
  def solve_all(self):
    solution = {group_name:self.solve_name(group_name)[0] for group_name in self.group_names}
    return solution

  def build_solver(self,member_num,expectation_member_capacity):
    X = Array.create('X', (self.department_num,self.term_num, member_num), 'BINARY')
    # Term_Member Once Constraint
    C_TM_Once = 0.0
    for t in range(self.term_num):
      for m in range(member_num):
        C_TM_Once += Constraint(
            (Sum(start_index=0,end_index=self.department_num,func=lambda d:X[d,t,m]) - 1)**2
            , label=f"TM_Once_{t},{m}") 
    # Department_Member Once Constraint
    C_DM_Once = 0.0
    for d in range(self.department_num):
      for m in range(member_num):
        C_DM_Once += Constraint(
            (Sum(start_index=0,end_index=self.term_num,func=lambda t:X[d,t,m]) - 1)**2
            , label=f"DM_Once_{t},{m}") 
    # Member Capacity Constraint
    C_MC = 0.0
    for d in range(self.department_num):
      for t in range(self.term_num):
        C_MC += Constraint(
            (Sum(start_index=0,end_index=member_num,func=lambda m:X[d,t,m]) - expectation_member_capacity[d,t])**2
            , label=f"MC_{d},{t}") 

    P_TM_Once = Placeholder("PTMO")
    P_DM_Once = Placeholder("PDMO")
    P_MC = Placeholder("PMC")

    H = P_TM_Once*C_TM_Once + P_DM_Once*C_DM_Once + P_MC*C_MC
    model = H.compile()
    return model

  def solve(self,model,shape=None,feed_dict=None):
    if feed_dict is None:
      feed_dict = {'PTMO': 2,'PDMO': 2,'PMC': 1}
    qubo, offset = model.to_qubo(feed_dict=feed_dict)
    sol = solve_qubo(qubo)
    if shape is None:
      np_X = self.decode(sol)
    else:
      np_X = self.decode_(sol,shape)
    return np_X

  def solve_name(self,group_name,check = True):
    np_X = self.solve(self.model_dict[group_name],
                      shape=(self.department_num,self.term_num,self.group_num_dict[group_name]))
    result = True
    out_list = []
    if check:
      if not self.TM_test(np_X):
        result = False
        out_list.append("TM")
      if not self.DM_test(np_X):
        result = False
        out_list.append("DM")
      if not self.MC_test(np_X,self.region_dict[group_name]):
        result = False
        out_list.append("MC")
    return np_X,result,out_list
            
class PostProcess():
  def __init__(self,np_X,df,department_names,term_names,member_num):
    self.hist=[]
    self.df = df
    self.member_names = self.df.name
    self.member_num = member_num
    self.department_names = department_names
    self.department_num = len(department_names)
    self.term_names = term_names
    self.term_num = len(term_names)
    self.set_np_X(np_X)

  def set_np_X(self,np_X):
    self.np_X = np_X
    self.score = self.l2_score(self.np_X)
    self.np_MET = self.count_MET(self.np_X)
    self.np_MET_bi = self.count_MET(self.np_X,bi=True)
    self.MET_max = self.np_MET.max()
    self.df_program = self.make_program(self.np_X)
    self.hist.append(self.df_program.copy(deep=True))
   
  def get_max_indexes(self):
    I,J=np.where(self.np_MET==self.np_MET.max())
    return [(I[ind],J[ind]) for ind in range(len(I))]
  def get_indexes(self,np_X,val):
    I,J = np.where(np_X==val)
    return [(I[ind],J[ind]) for ind in range(len(I))]

  def encounter_sr(self,i):
    return self.np_MET_bi[i]

  def make_program(self,np_X):
    df_program = pd.DataFrame(index = self.department_names, columns = self.term_names)
    for d in range(self.department_num):
      for t in range(self.term_num):
        participant_ids=np.where(np_X[d,t]==1)[0]
        df_program.iat[d,t] = tuple([self.df.name[pid] for pid in participant_ids])
    return df_program
  def get_canditate_list(self,pind,npind):
    ref_ind = ((pind[0],npind[1]),(npind[0],pind[1]))
    v1 = self.np_X[pind[0],npind[1]]
    v2 = self.np_X[npind[0],pind[1]]
    return np.where(v1*v2==1)[0]

  def extra_check(self,canditate,rep):
    #implement here
    return canditate
  def search_canditate(self,pair,replace):
    i,j = pair
    rep = replace
    rep_sr = self.encounter_sr(rep)

    paired = self.np_X[:,:,i]*self.np_X[:,:,j]
    rep_not_paired = self.np_X[:,:,rep] - self.np_X[:,:,i] * self.np_X[:,:,j]
    paired_indexes = self.get_indexes(paired, 1)
    not_paired_indexes = self.get_indexes(rep_not_paired,1)
    canditate = {}
    for pind in paired_indexes:
      for npind in not_paired_indexes:
        temp_canditate=self.get_canditate_list(pind,npind)
        for tcan in temp_canditate:
          canditate[tcan] = (pind,npind)

    canditate = self.extra_check(canditate,rep)
    canditate_count = np.array([rep_sr[can] for can in canditate.keys()])

    result = {}
    result["success"] = (len(canditate) > 0)
    if result["success"]:
      result["member_index"] = list(canditate.keys())[canditate_count.argmin()]
      result["count"] = canditate_count.min()
      result["replace_DTs"] = canditate[result["member_index"]] # i
    return result

  def replace(self,c,i):
    i_ind = i
    i_DTs = c["replace_DTs"]
    r_ind = c["member_index"]
    r_DTs = ((i_DTs[0][0],i_DTs[1][1]),(i_DTs[1][0],i_DTs[0][1]))
    temp_X = self.np_X.copy()
    for DT in i_DTs:
      if temp_X[DT[0],DT[1],i_ind] == 1 and temp_X[DT[0],DT[1],r_ind] == 0:
        # confirmed 
        temp_X[DT[0],DT[1],i_ind] = 0
        temp_X[DT[0],DT[1],r_ind] = 1
      else:
        raise Exception # something wrong

    for DT in r_DTs:
      if temp_X[DT[0],DT[1],i_ind] == 0 and temp_X[DT[0],DT[1],r_ind] == 1:
        # confirmed 
        temp_X[DT[0],DT[1],i_ind] = 1
        temp_X[DT[0],DT[1],r_ind] = 0
      else:
        raise Exception # something wrong
    return temp_X

  def count_MET(self,np_X,bi=False):
    #member_num = np_X.shape[2]
    member_num = self.member_num
    np_MET = np.zeros((member_num,member_num),dtype=int)
    if bi:
      for m_i in range(member_num):
        for m_j in range(member_num):
          v_i = np_X[:,:,m_i]
          v_j = np_X[:,:,m_j]
          np_MET[m_i,m_j] = np.sum(v_i*v_j)
      for m in range(member_num):
        np_MET[m,m] = 2
    else:
      for m_i in range(member_num):
        for m_j in range(m_i):
          v_i = np_X[:,:,m_i]
          v_j = np_X[:,:,m_j]
          np_MET[m_i,m_j] = np.sum(v_i*v_j)
    return np_MET

  def l2_score(self,np_X):
    np_MET = self.count_MET(np_X)
    return np.sum(np_MET**2)

  def main(self,show=True):
    go_flag = True
    trial = 0
    while go_flag:
      trial += 1
      go_flag = False
      query = self.get_max_indexes()
      vmax = self.np_MET.max()

      if show:
        print()
        print(f"trial:{trial}")
        print(query,vmax,end="\n\n")
      for (i,j) in query:
        c_i = self.search_canditate((i,j),i)
        c_j = self.search_canditate((i,j),j)
        if c_i["success"] or c_j["success"]:
          # success
          if not c_i["success"]*c_j["success"]:
            if c_i["success"]:
              np_New_X=self.replace(c_i,i)
              replaced_ind = (i,c_i["member_index"])
              cc = c_i
            else:
              np_New_X=self.replace(c_j,j)
              replaced_ind = (j,c_j["member_index"])
              cc = c_j
          else:
            if c_i["count"] < c_j["count"]:
              np_New_X=self.replace(c_i,i)
              replaced_ind = (i,c_i["member_index"])
              cc = c_i
            else:
              np_New_X=self.replace(c_j,j)
              replaced_ind = (j,c_j["member_index"])
              cc = c_j
          # confirm score    
          if self.score > self.l2_score(np_New_X):
            self.set_np_X(np_New_X)
            if show:
              print(f"to {(i,j)}")
              print(f"replaced {replaced_ind}")
              count = cc["count"]
              print(f"count {vmax}->{count}")
              print("replaced D,M index {}".format(cc["replace_DTs"]))
              print(f"score {self.l2_score(np_New_X)}")
            go_flag = True
        else:
          #fail
          pass
    if show:
      print("postprocessing ended")
