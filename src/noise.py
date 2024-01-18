import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        Initialize the Noise object.

        Parameters:
            - mean (float): The mean value of the noise.
            - std_deviation (float): The standard deviation of the noise.
            - theta (float, optional): The theta parameter for the Ornstein-Uhlenbeck process. Default is 0.15.
            - dt (float, optional): The time step for the Ornstein-Uhlenbeck process. Default is 1e-2.
            - x_initial (float, optional): The initial value of the noise. Default is None.

         Returns:
            - None
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
        Generate noise using the Ornstein-Uhlenbeck process.

        Parameters:
            - None

        Returns:
            - x (numpy.ndarray): The generated noise.
        """
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the noise process.
        If `x_initial` is provided, sets `x_prev` to `x_initial`.
        Otherwise, sets `x_prev` to an array of zeros with the same shape as `mean`.

        Parameters: 
            - None

        Returns:
            - None
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
