from .grader import Grader, Case

import pystk
import numpy as np

class TuxKartGrader(Grader):
    """Driving in SupertuxKart"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        drive = self.module.drive
        
        self.race = None
        self.track = None
        
        self._config_race()

        self.scores = []
        for _ in range(10):
            self.scores.append(self._race(drive, 'battleisland'))
        
    @Case(score=100)
    def _grade(self):
        """battleisland"""
        return np.mean(self.scores)
        
    def _config_race(self):
        g_config = pystk.GraphicsConfig.hd()
        g_config.screen_width = 160
        g_config.screen_height = 120
        
        pystk.init(g_config)
        
        
    def _race(self, drive, track_name, time_limit=1200):
        
        total_length = 1.0

        score = 0.

        try:
            config = pystk.RaceConfig(num_kart=6, track=track_name, mode=pystk.RaceConfig.RaceMode.FREE_FOR_ALL)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            config.step_size = 0.1
            
            race = pystk.Race(config)
            race.start()
            race.step()

            
            for t in range(time_limit):
                
                state = pystk.WorldState()
                state.update()

                scores = state.ffa.scores
                kart = state.players[0].kart
                
                action = drive(np.asarray(race.render_data[0].image), kart)
                
                race.step(action)

            rank = sorted(scores, reverse=True).index(scores[kart.id])
            score = {0:10,1:8,2:6}.get(rank, 7-rank)
            
        finally:
            race.stop()
            del race

        return score/10.
