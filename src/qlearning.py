import numpy as np
import pandas as pd
import time

#np.random.seed(2)

N_STATES = 12 #the length of the 1 dimensial world
ACTIONS = ['left','right']
EPSILON = 0.9 #greedy ploicy
ALPHA = 0.1 #learining rate
LAMBDA = 0.9 #discount 
MAX_EPISODES = 13 #
FRESH_TIME = 0.3 

def build_Qtable(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions)
    print(table)
    return table

def chose_Action(state, q_table):
    state_actions = q_table.iloc[state,:]
    #print('uniform=',np.random.uniform())
    
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.idxmax()
            
    return action

def get_env_feedback(state,A):
    R = 0
    if A == 'right':
        if state == N_STATES - 2:
            state_ = 'terminal'
            R = 1
        else:
            state_ = state + 1
    else:
        if state == 0:
            state_ = 0
        else:
            state_ = state - 1
    return state_,R

def updateEnv(S,episode,step_counter):
    env_list=['-']*(N_STATES-1) + ['T'] # '------T'
    if S == 'terminal':
        interaction = 'Episode %s:total_steps=%s' % (episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\n')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list) + ' Episode ' + str(episode)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)
        
def main():
    q_table = build_Qtable(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter=0
        S=0
        isTerminated = False
        updateEnv(S,episode,step_counter)
        while not isTerminated:
            A = chose_Action(S,q_table)
            S_, R = get_env_feedback(S,A)
            #print('A=',A)
            q_predict = q_table.loc[S,A] 
            
            if S_ != 'terminal':
                q_target = R + LAMBDA* q_table.iloc[S_, :].max()
            else:
                q_target = R
                isTerminated = True
                
            q_table.loc[S, A] += ALPHA*(q_target-q_predict)
            #print(q_table)
            S = S_
            updateEnv(S,episode,step_counter+1)
            step_counter += 1
            
        print(q_table)    

if __name__=='__main__':
    main()
    