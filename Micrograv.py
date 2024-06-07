import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math

# Constants
LENGTH = 0.06  # Length of the box along x-axis
WIDTH = 0.06   # Width of the box along y-axis
HEIGHT = 0.05  # Height of the box along z-axis
NUM_PARTICLES = 100  # Number of particles
NUM_STEPS = 50      # Number of simulation steps
dt = 0.05             # Time step

v_sound = 343  # Speed of sound in air (in meters per second)

# Calculate wavelength of fundamental frequency
wavelength = 2 * LENGTH

# Calculate wavelength of second harmonic (n=2)
wavelength_2nd = wavelength / 2

# Calculate frequency of second harmonic
f_2nd = v_sound / wavelength_2nd
#f_2nd = 
print(f_2nd)


# Initialize particles randomly within the box
particles = np.random.rand(NUM_PARTICLES, 3) * np.array([LENGTH, WIDTH, HEIGHT])
particles[-1] = [0,0,HEIGHT/2]
#particles[-1] = [LENGTH/2,WIDTH/2,HEIGHT/2]
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

def is_within_10_percent(point, reference_point, percent, length, width):
    """
    Check if a point (x, y) is within 10% of the reference_point (rx, ry)
    considering the length and width dimensions.
    """
    x, y = point
    rx, ry = reference_point
    
    # Calculate 10% of length and width
    ten_percent_length = length * percent
    ten_percent_width = width * percent
    
    # Calculate the lower and upper bounds for x and y
    lower_bound_x = rx - ten_percent_width
    upper_bound_x = rx + ten_percent_width
    lower_bound_y = ry - ten_percent_length
    upper_bound_y = ry + ten_percent_length
    
    # Check if the point is within the bounds
    return lower_bound_x <= x <= upper_bound_x and lower_bound_y <= y <= upper_bound_y

def check_point_within_locations(point, specified_locations, length, width, percent, individual_locations):
    """
    Check if a point is within 10% of any specified locations.
    """
    for i, location in enumerate(specified_locations):
        
        if is_within_10_percent(point, location, percent, LENGTH, WIDTH):
            individual_locations[i] = individual_locations[i]+1
            return True, individual_locations
    return False, individual_locations


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

    #count_above_zero.append(len([particle for particle in particles if particle[2] > .02]))
    count = 0
    spec_count = 0
    central_count = 0
    specified_locations = {(0,0), (0,LENGTH/2), (0, LENGTH), (WIDTH/2, 0), (WIDTH/2, LENGTH/2), (WIDTH/2, LENGTH), (WIDTH, 0), (WIDTH, LENGTH/2), (WIDTH, LENGTH)}
    #MAKE SURE SET IS SORTED BECAUSE ORDER CAN CHANGE ON ITERATION
    sorted_specified_locations = sorted(specified_locations)
    individual_particle_locations = [0]*9
    percent = .1
    for particle in particles:
        if particle[2] > 0.02:
            count += 1
        Is_in_nodal, individual_particle_locations_placeholder = check_point_within_locations(particle[:2], sorted_specified_locations, LENGTH, WIDTH, percent, individual_particle_locations)
        if Is_in_nodal:
            spec_count+=1
        
        if is_within_10_percent(particle[:2], (LENGTH/2, WIDTH/2), percent, LENGTH, WIDTH):
            central_count += 1
        #if tuple(particle[:2]) in specified_locations:
            #spec_count += 1

        
    count_above_zero.append(count)
    count_spec.append(spec_count)
    count_central.append(central_count)
    individual_count1.append(individual_particle_locations[0])
    individual_count2.append(individual_particle_locations[1])
    individual_count3.append(individual_particle_locations[2])
    individual_count4.append(individual_particle_locations[3])
    individual_count5.append(individual_particle_locations[4])
    individual_count6.append(individual_particle_locations[5])
    individual_count7.append(individual_particle_locations[6])
    individual_count8.append(individual_particle_locations[7])
    individual_count9.append(individual_particle_locations[8])


    #print(count_above_zero)

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

count_above_zero = list()
count_spec = list()
count_central = list()
individual_count1 = list()
individual_count2 = list()
individual_count3 = list()
individual_count4 = list()
individual_count5 = list()
individual_count6 = list()
individual_count7 = list()
individual_count8 = list()
individual_count9 = list()

# Create animation
ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=20)

#time = [dt * i for i in range(1, runtime+1)]
#print(len(time))
all_location_lists = [individual_count1, individual_count2, individual_count3,individual_count4,individual_count5,individual_count6,individual_count7,individual_count8, individual_count9]


plt.show()

#print(len(count_above_zero))
time = [dt*i for i in range(0, len(count_above_zero))]
#print(len(time))
plt.figure(2)
plt.plot(time, count_above_zero)
plt.ylabel("Number of particles above zero")
plt.xlabel("Time (sec)")
plt.title(f"Number of particles above ground over time ({NUM_PARTICLES} Particles)")
plt.show()

plt.figure(3)
plt.plot(time, count_spec)
plt.ylabel("Number of particles in nodes")
plt.xlabel("Time (sec)")
plt.title(f"Number of particles in nodal lines over time ({NUM_PARTICLES} Particles)")
plt.show()

plt.figure(3)
plt.plot(time, count_central)
plt.ylabel("Number of particles in central node")
plt.xlabel("Time (sec)")
plt.title(f"Number of particles in the central node over time ({NUM_PARTICLES} Particles)")
plt.show()

fig2, ax2 = plt.subplots(3,3, figsize = (10,10))
ax2 = ax2.flatten()
for i, data in enumerate(all_location_lists, start=0):
    ax2[i].plot(time, data)
    ax2[i].set_title(f"Node {i+1}")
    ax2[i].set_xlabel("Time (sec)")
    ax2[i].set_ylabel("Number of particles in Node")

fig2.suptitle(f'Number of particles in each individual Node ({NUM_PARTICLES} Particles)', fontsize=16)
plt.tight_layout()
plt.show()

#todo - find out why node 1 has no particles in it