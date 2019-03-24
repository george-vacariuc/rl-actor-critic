import traceback, sys

# A base for runnable tasks.
class Runnable():
    def __init__(self):
        self._stop = False
        self._ticks = 0

    def run(self):
        try:
            print('### run #  Entered.')
            self._run()
        except:
            traceback.print_exc(file=sys.stdout)
        
    def stop(self):
        self._stop = True

    @property
    def ticks(self):
        return self._ticks
    
    def tick(self):
        self._ticks += 1

        
                
        