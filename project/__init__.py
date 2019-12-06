import pystk
import gym, ray
import numpy as np
from ray.rllib.agents import ppo

class TuxEnv(gym.Env):
    def __init__(self, _):
        print("calling __init__")
        self.action_space = gym.spaces.Box(np.array([-1,0,0,0]), np.array([1,1,0,0]))  # steer, fire
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(120, 160,3))

    def reset(self):
        print("reset?")
        pass

    def step(self, action):
        print("step?????")
        pass

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
        self.img = None
        self.action = 0

    def get_img(self):
        return self.img

    def set_img(self, img):
        self.img = img

    def set_action(self, action):
        self.action = action

    def increment(self):
        self.value += 1
        return self.value, self.action

ray.init(ignore_reinit_error=True)

count = Counter.remote()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 0
config["eager"] = False
agent = ppo.PPOAgent(config=config, env=TuxEnv)
agent.restore("/content/gdrive/My Drive/checkpoint-1")

prev_img = None

def drive(img, kart:pystk.Kart):
    """
    @img: (120,160,3) RGB image
    return: pystk.Action
    """
    img = np.asarray(img) / 255.0

    i, old_action  = ray.get(count.increment.remote())
    print(i, old_action)
    prev = ray.get(count.get_img.remote())

    if type(old_action) == int:
        old_action = [old_action,old_action]
    if i % 5 == 0:
        action = agent.compute_action(img)
    else:
        action = old_action

    steer_dir = action[0]
    #gas = action[1]
    #brake = action[2]
    fire = action[1]
    setter = count.set_img.remote(img)
    count.set_action.remote(action)

    if i > 10:
        if np.sum(np.abs(img-prev)) < 50:
            return pystk.Action(rescue=True)

    return pystk.Action(steer=steer_dir, acceleration=1.0, fire=fire)