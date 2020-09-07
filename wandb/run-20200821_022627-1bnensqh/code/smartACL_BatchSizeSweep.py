# Train an agent from scratch with PPO2 and save package and learning graphs
# from OpenGL import GLU
import os
import glob
import time
import subprocess
import shutil
import gym
import wandb
import random
import logging
from collections import defaultdict
from gym_smartquad.envs import quad_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import customMonitor
import datetime
import imageio
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
class smartCurriculumCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: no output 1: info 2: debug
    """
    def __init__(self, envFunct, refreshRate, betaRate, initCurriculum, endDomain, targetDomain, domainPowers, ADRMethod = 'ep_reward_mean',targetReliability=None, targetReward=None, renders = True, verbose=1):
        super(smartCurriculumCallback, self).__init__(verbose)
        self.refreshRate = refreshRate
        self.evalRate = 100000
        self.betaRate = betaRate

        self.n_calls = 0

        self.oldLoss = None
        self.newLoss = None
        self.oldStep = 0
        self.oldEvalStep = 0

        self.meanRew = 0
        self.rewardScale = 700 #TO BE FULLY INTEGRATED

        self.envFunct = envFunct

        self.curriculum = initCurriculum
        self.initCurriculum = initCurriculum
        self.endDomain = endDomain
        self.progress = 0

        self.targetDomain = targetDomain
        self.domainPowers = domainPowers

        self.targetReliability = targetReliability
        self.targetReward = targetReward

        self.best_min = np.NINF
        self.best_mean = np.NINF

        self.ADRMethod = ADRMethod
        self.n_eval_episodes = 15
        self.evaluations_results = []
        self.evaluations_mins = []
        self.evaluations_timesteps = []
        self.gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_gif/")
        self.models_tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models_tmp/")
        self.renders = renders
        # Those variables will be accessible in the callback
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        self.logger_dir = None
        #self.logger_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logger/")
        #logger.make_output_format('csv', self.logger_dir, log_suffix = 'progresslog')
        

    def _on_training_start(self) :
        """
        This method is called before the first rollout starts.
        """
        self.logger_dir = self.logger.get_dir() +'/'+ os.listdir(self.logger.get_dir())[0]

    def _on_rollout_start(self) :
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) :

        #Basic curriculum progression, every self.refreshRate timesteps
        if self.num_timesteps-self.oldStep >= self.refreshRate  :
            self.oldStep = self.num_timesteps
            print(self.num_timesteps)
            #Loss based ADR
            if self.ADRMethod == 'loss':
                self.lossMethod()
            
            if self.ADRMethod == 'ep_reward_mean':
                self.rewardMethod()

        #evaluation 
        if self.num_timesteps - self.oldEvalStep >= self.evalRate :
            self.oldEvalStep = self.num_timesteps

            evalEnv = self.envFunct(self.targetDomain)
            #sync_envs_normalization(self.training_env, evalEnv)
            episode_rewards, episode_lengths = evaluate_policy(self.model, evalEnv,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               return_episode_rewards=True)
            print(episode_rewards)
            self.evaluations_results.append(np.mean(episode_rewards))
            self.evaluations_mins.append(np.min(episode_rewards))
            self.evaluations_timesteps.append(self.num_timesteps)

            #remembering the best results :  
            if np.mean(episode_rewards) == np.max(self.evaluations_results):
                self.best_mean =  np.mean(episode_rewards)
                self.best_min = np.min(episode_rewards)
                #wandb.log({"best_mean": self.best_mean, "best_min": self.best_min}, step=self.num_timesteps)
                self.model.save(self.models_tmp_dir +"best_network"+".plk")


            #TO IMPLEMENT : SAVE THE NETWORK

            #External logging
            wandb.log({"eval_reward": self.evaluations_results[-1]}, step=self.num_timesteps) 
            wandb.log({"best_mean": self.best_mean, "best_min": self.best_min}, step=self.num_timesteps)
            print('average score in a real environment : '+str(self.evaluations_results[-1]))
            print('minimum score in a real environment : '+str(self.evaluations_mins[-1]))
            self.model.save(self.models_tmp_dir +"step_"+str(self.num_timesteps)+".plk")

            if self.renders :
                self.createGif(evalEnv)
            else :
                evalEnv.close()
            #Not used yet
            if self.targetReliability!=None and self.targetReward!=None:
                goalReached = True
                for i in range(self.n_eval_episodes):
                    if episode_rewards < self.targetReward :
                        goalReached = False
                
                if goalReached :
                    return False  
            
        return True

    def rewardMethod(self):   
        summary_iterators = EventAccumulator(self.logger_dir).Reload() 
        tags = summary_iterators.Tags()['scalars']
        out = defaultdict(list)
        for tag in tags:
            #steps = [e.step for e in summary_iterators.Scalars(tag)]
            for events in summary_iterators.Scalars(tag):
                out[tag].append([e for e in events])     
        out = np.array(out['rollout/ep_rew_mean'])
        #print(out)   #May help debugging in case anything happens
        try :
            self.meanRew = out[-1,2]
        except : #if there is only one logged element
            try :
                self.meanRew = out[2]
            except : #if nothing is logged yet
                return True

        print(self.curriculum)
        for i in range(len(self.curriculum)): 
            self.progress = self.meanRew/self.rewardScale
            if self.progress < 0 :
                self.progress = 0
            elif self.progress > 1 :
                self.progress = 1

            #For now, the only supported progression goes from the simplest to the most difficult
                
            self.curriculum[i] = self.initCurriculum[i] + (self.endDomain[i]-self.initCurriculum[i])*self.progress**self.domainPowers[i]
            #print(self.progress)
        self.training_env.env_method('refresh',self.curriculum)
        wandb.log({"domain_progress": self.progress}, step=self.num_timesteps)


    def lossMethod(self):   
        summary_iterators = EventAccumulator(self.logger_dir).Reload() 
        tags = summary_iterators.Tags()['scalars']
        out = defaultdict(list)
        for tag in tags:
            #steps = [e.step for e in summary_iterators.Scalars(tag)]
            for events in summary_iterators.Scalars(tag):
                out[tag].append([e for e in events])     
        out = np.array(out['train/loss'])
        #print(out)   #May help debugging in case anything happens
        try :
            meanLoss = out[:,2]
        except : #if there is only one logged element
            try :
                meanLoss = out[2]
            except : #if nothing is logged yet
                return True
        
        try :
            meanLoss = np.mean(meanLoss[-5:])  #may be edited
        except :
            meanLoss = meanLoss[-1]
            
        if self.oldLoss != None :
            self.oldLoss = self.newLoss
            self.newLoss = meanLoss
   
            lossDiff = self.newLoss-self.oldLoss
            #Updating the curriculum

            if lossDiff > 0 :
                print(self.curriculum)
                for i in range(len(self.curriculum)): 
                    progressStep = self.betaRate*lossDiff
                    #Clipping progress :
                    if progressStep > 0.05 :
                        progressStep = 0.05 
                    self.progress += progressStep 

                    #For now, the only supported progression goes from the simplest to the most difficult
                    if self.progress>1 :
                        self.progress=1
                    self.curriculum[i] = self.initCurriculum[i] + (self.endDomain[i]-self.initCurriculum[i])*self.progress**self.domainPowers[i]
                    #print(self.progress)
                self.training_env.env_method('refresh',self.curriculum)
            wandb.log({"domain_progress": self.progress, "loss_dif": lossDiff}, step=self.num_timesteps)
            print(self.num_timesteps)
        else : 
            self.newLoss = meanLoss
            self.oldLoss = self.newLoss


    def createGif(self,evalEnv):
        gif_name = "PPO_"+str(self.num_timesteps)
        save_str = self.gif_dir + gif_name + '.gif'


        model = PPO.load(self.models_tmp_dir +"step_"+str(self.num_timesteps)+".plk", env=evalEnv)
        images = []
        obs = evalEnv.reset()
        img = evalEnv.sim.render(
            width=400, height=400, camera_name="isometric_view")
        for _ in range(600):
            action, _ = model.predict(obs)
            obs, _, _, _ = evalEnv.step(action)
            img = evalEnv.sim.render(
                width=400, height=400, camera_name="isometric_view")
            images.append(np.flipud(img))
        #print("creating gif...")
        imageio.mimsave(save_str, [np.array(img)
                                   for i, img in enumerate(images) if i % 2 == 0], fps=29)
        print("gif created...")
        evalEnv.close()

    def _on_rollout_end(self) :
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) :
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class PPOtraining():
    def __init__(self, envFunct, trainingSteps, targetDomain, domainPowers, domainProgressRate, learningRate = 0.0003, batchSize = 256, ADRMethod ='loss', autoParameters = False, startDomain=None, endDomain=None, targetReliability=None, targetReward=None, initModelLoc=None, render=False, verbose = 1, tag="")  :
        """Trains a model using PPO (baselines3, based on PyTorch). 

        Env : 
            Must be imported at the beginning of the file, and declared in the 'updateEnv' method. Please do check out the evaluation section of the Callback class. Currently supports Gym structure.
            WARNING : In order to support smart curriculum learning, the environment must incorporate a 'refresh' method, which updates domains parameters. 
            Note that this additionnal method is different from a reset method, but can simply update values that will be used in the 'reset' method (especially true for geometrical parameters).
            As for noise parameters, they can be directly used after being updated, even in the middle of an episode.


        Args:
            envFunct : function. See the aforementioned 'Env' section.
            trainingSteps : int. Total number of steps for the training. 
            autoParameters : bool. False by default. Automatically assess the impact of domain variable on the performance of the neural network, and infers custom progress parameters for the ADR.
            targetDomain : vector (1D) of domain parameters estimated as representative of the real environment. If possible, it is recommended to characterize such values by performing measurements of the sub-systems (sensors/actuators), or environmental parameters.
                           These parameters can also be infered from the system requirements. Set to a Null vector to ignore Domain Randomization.
            domainPowers : same dimension as targetDomains. Vector of empirical parameters. Default should be np.ones(targetDomain.shape).
                            1 means that that this parameter will be more or less linearly increased throughout the learning. x<1 means that the parameter will mostly increase in the final phase of the learning. x>1 means that that parameter will mostly increase in the early phase of the learning.
                            Base function is parameters = (progress)**domainPowers, with progress belonging in [0,1].   Set to a 0.00001 vector to ignore Curriculum Learning.
            domainProgressRate : float < 1. Describes of fast is the ADR going to progress. Requires a bit of fine-tuning. Set such as the domain_progress reaches 1 toward the end of the training. A uniform parameter sweep is probably the best best way to go.
        KwArgs :
            startDomain : vector (1D) of domain parameters to begin the learning with. None by default, meaning that all of these parameters will be 0.
            endDomain : vector (1D) of domain parameters to end the learning with. By default, equals to None, and is automatically chosen as being equal to targetDomain.
            targetReliability : float in [0,1]. Enables validation to be performed every now and then, and the learning process to be stopped when the model achieves a targetReliability rate of success (achieving targetReward with an environment defined with targetDomain) 
            targetReward : float. 
            initModelLoc : path to a stable_baselines model. Enables a previously trained model to be improved with domain randomization
            render : bool. Default is 0. Renders Gifs of the target environment every 100000 steps.
            verbose : bool. Default is 1. Display essential learning data in the shell.
            tag : str. fefault is "". Puts a label on the best saved network 
        """
  
        self.step_total = trainingSteps
        self.verbose = verbose
        self.env = None
        self.envFunct = envFunct
        self.n_cpu = 8
        self.modelLoc = initModelLoc
        self.model = None


        self.batchSize = batchSize
        self.learningRate = learningRate

        self.tag = tag
        self.createDirectories()

        #Callbacks parameters
        self.refreshRate = 30000
        self.betaRate = domainProgressRate
        self.targetDomain = targetDomain
        self.domainPowers = domainPowers
        if not isinstance(startDomain,np.ndarray) :
            self.curriculum = np.zeros(targetDomain.shape)
        else :
            self.curriculum = startDomain
        if not isinstance(endDomain,np.ndarray) :
            self.endDomain = self.targetDomain
        else :
            self.endDomain = endDomain
        self.renders = render
        self.ADRMethod = ADRMethod 
        #External logging
        

        self.updateEnv(self.curriculum)
        self.updateModel()
        self.train()
        wandb.join()
        


    def train(self):
        start = time.time()

        #evalEnv = self.envFunct(self.targetDomain)
 
        #self.model.learn(total_timesteps=self.step_total, eval_env = evalEnv, eval_freq = 20000, n_eval_episodes= 15,log_interval=1, tb_log_name="PPO",callback=smartCurriculumCallback(self.refreshRate, self.betaRate, self.curriculum, self.targetDomain, self.domainPowers, targetReliability=None, targetReward=None, renders = False, verbose=1))
        #Using callbacks to perform evaluations instead :
        callbackFunction = smartCurriculumCallback(self.envFunct, self.refreshRate, self.betaRate, 
                                                    self.curriculum, self.endDomain, self.targetDomain, 
                                                    self.domainPowers, ADRMethod = self.ADRMethod, targetReliability=None, targetReward=None, 
                                                    renders = self.renders , verbose=1)
        self.model.learn(total_timesteps=self.step_total,log_interval=1, tb_log_name="PPO",callback=callbackFunction)
        end = time.time()
        training_time = end - start 
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models_tmp/best_network.plk")
        dst_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/best_network"+wandb.run.id+self.tag+timestamp+".plk")
        shutil.copy(src_dir,dst_dir)
        
        #Performance summary is now handled by the validation log.

      
    def updateEnv(self, initCurriculum):

        if self.env != None :
            self.env.close()
        
        self.env = self.envFunct(initCurriculum)
        self.env = customMonitor.Monitor(self.env, allow_early_resets=True)
        self.env = DummyVecEnv( [lambda: self.env for i in range(self.n_cpu)] )
        

    def updateModel(self):
        if self.modelLoc==None: 
            self.model = PPO(MlpPolicy, self.env, tensorboard_log="./logger/",verbose=1, device='cuda',n_steps = 2048, n_epochs=10, batch_size= self.batchSize, learning_rate= self.learningRate)
            self.modelLoc = self.models_dir
        else :
            pass


    def createDirectories(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/")
        
        self.models_tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models_tmp/")
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        self.gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_gif/")
        self.plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot")
        self.logger_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logger/")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.models_tmp_dir, exist_ok=True)
        os.makedirs(self.plt_dir, exist_ok=True) 
        os.makedirs(self.logger_dir, exist_ok=True) 

if __name__ == '__main__':
    #CArefull set modified
    for i in range(5):
        params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paramsADR.json")
        with open(params_path) as json_file:
            params = json.load(json_file)
        tag = params["tag"]
        wandb.init(project="Smart_Quad_Friction_wide", sync_tensorboard=True, allow_val_change=True, reinit=True, tags=[tag])
        wandb.config.progress_rate = params["progress_rate"]
        wandb.config.domain_powers = None
        wandb.config.learningRate = 0.002
    
        wandb.config.batchSize = 1800
    
        training = PPOtraining(quad_env.QuadEnv, 2000000,  np.array(params["targetDomain"]), np.array(params["domainPowers"]), wandb.config.progress_rate , learningRate = wandb.config.learningRate, batchSize = wandb.config.batchSize, startDomain= np.array(params["startDomain"]), endDomain =  np.array(params["endDomain"]), ADRMethod = 'loss', targetReliability=None, targetReward=None, initModelLoc=None, render = False, verbose = 1, tag = tag)
   
    for i in range(5):
        params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paramsDR.json")
        with open(params_path) as json_file:
            params = json.load(json_file)
        tag = params["tag"]
        wandb.init(project="Smart_Quad_Friction_wide", sync_tensorboard=True, allow_val_change=True, reinit=True, tags=[tag])
        wandb.config.progress_rate = params["progress_rate"]
        wandb.config.domain_powers = None
        wandb.config.learningRate = 0.002
    
        wandb.config.batchSize = 1800
    
        training = PPOtraining(quad_env.QuadEnv, 2000000,  np.array(params["targetDomain"]), np.array(params["domainPowers"]), wandb.config.progress_rate , learningRate = wandb.config.learningRate, batchSize = wandb.config.batchSize, startDomain= np.array(params["startDomain"]), endDomain =  np.array(params["endDomain"]), ADRMethod = 'loss', targetReliability=None, targetReward=None, initModelLoc=None, render = False, verbose = 1, tag = tag)

    for i in range(5):
        params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paramsNoDR.json")
        with open(params_path) as json_file:
            params = json.load(json_file)
        tag = params["tag"]
        wandb.init(project="Smart_Quad_Friction_wide", sync_tensorboard=True, allow_val_change=True, reinit=True, tags=[tag])
        
        wandb.config.progress_rate = params["progress_rate"]
        wandb.config.domain_powers = None
        wandb.config.learningRate = 0.002
    
        wandb.config.batchSize = 1800
    
        training = PPOtraining(quad_env.QuadEnv, 2000000,  np.array(params["targetDomain"]), np.array(params["domainPowers"]), wandb.config.progress_rate , learningRate = wandb.config.learningRate, batchSize = wandb.config.batchSize, startDomain= np.array(params["startDomain"]), endDomain =  np.array(params["endDomain"]), ADRMethod = 'loss', targetReliability=None, targetReward=None, initModelLoc=None, render = False, verbose = 1, tag = tag)

