import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constants
LENGTH = 0.05  # Length of the box along x-axis
WIDTH = 0.05   # Width of the box along y-axis
HEIGHT = 0.06  # Height of the box along z-axis
NUM_PARTICLES = 1000  # Number of particles
NUM_STEPS = 50      # Number of simulation steps
dt = 0.05             # Time step

v_sound = 343  # Speed of sound in air (in meters per second)

# Calculate wavelength of fundamental frequency
wavelength = 2 * LENGTH

# Calculate wavelength of second harmonic (n=2)
wavelength_2nd = wavelength / 2

# Calculate frequency of second harmonic
f_2nd = v_sound / wavelength_2nd
print(f_2nd)


# Initialize particles randomly within the box
particles = np.random.rand(NUM_PARTICLES, 3) * np.array([LENGTH, WIDTH, HEIGHT])
particles[-1] = [5.,5.,5.]
#Initialise particles with some random velocity
velocities = np.random.normal(0, 0.1, size=(NUM_PARTICLES, 3))
#particles = np.asarray([[5.,0.,5.]]) #y,x,z coords

# Initialize speakers
speaker1_pos = np.array([LENGTH / 2, 0, HEIGHT / 2])  # Speaker on the x-axis
speaker2_pos = np.array([0, WIDTH / 2, HEIGHT / 2])    # Speaker on the y-axis


# Function to calculate standing wave displacement
#This just calculates the y value (amplitude) of each wave at the position given
def standing_wave_displacement(point, speaker_pos,speaker_pos_2,  time):

    #Calculates the distance from each speaker
    distance_to_speaker1 = point[1]
    distance_to_speaker2 = point[0]

    #Calculates the amplitude of the sound wave at each particle 
    wave1 = np.sin(2 * np.pi * distance_to_speaker1 / wavelength_2nd)
    wave2 = np.sin(2 * np.pi * distance_to_speaker2 / wavelength_2nd)
    #return wave1+wave2
    return wave1, wave2

# Function to update particle positions
def update(frame):
    global particles
    # Calculate displacement for each particle
    for i in range(NUM_PARTICLES):

        #Calculates the amplitude of speaker wave at each particle
        amp1, amp2 = standing_wave_displacement(particles[i], speaker1_pos, speaker2_pos,  frame * dt)

        #Update the position of the particles by adding on velocity*time step*amplitude of the waves
        particles[i, 0] += velocities[i, 0]*amp2*dt #can add dt it doesnt matter
        particles[i, 1] += velocities[i, 1]*amp1*dt
        particles[i, 2] += velocities[i, 2]*(amp1+amp2)*dt

    #Check boundaries and reflect particles if they hit the wall
    for j in range(3):
        # Check lower bound
        crossed_lower = particles[:, j] < 0
        particles[crossed_lower, j] = 0
        velocities[crossed_lower, j] *= -1  # Reverse velocity
        
        # Check upper bound
        crossed_upper = particles[:, j] > 10
        particles[crossed_upper, j] = 10
        velocities[crossed_upper, j] *= -1  # Reverse velocity

    # Boundary conditions (particles should stay within the box)
    particles = np.clip(particles, 0, [LENGTH, WIDTH, HEIGHT])

    # Update scatter plot
    scatter.set_offsets(particles[:, :2])
    scatter.set_3d_properties(particles[:, 2] , zdir='z')

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
scatter = ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='b', marker='o')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_xlim(0,LENGTH)
ax.set_ylim(0,WIDTH)
ax.set_zlim(0,HEIGHT)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Particle Simulation')

# Create animation
ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=20)

plt.show()
