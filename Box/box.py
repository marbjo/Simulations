import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D




class Particle():
    """Particle class with simple Newtonian physics"""

    #Constants
    box_size = 10 #Size of particle box (From center at origin)
    
    
    def __init__(self, x, y, z, vx, vy, vz, ax, ay, az, m, me, sz):
        self.x = x
        self.y = y
        self.z = z
        self.r = np.array((x, y, z))

        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.v = np.array((vx, vy, vz))
        
        self.ax = ax
        self.ay = ay
        self.az = az
        self.a = np.array((ax, ay, az))

        #if not dim_3:
        #    self.r[1], self.v[1], self.a[1] = 0 #Ensures y-component not used if 2D (gravity in z-direction)

        self.m = m
        self.me = me #Particle number (Particle knows its own "ID")
        self.sz = sz #Size (radius)

    def move(self, dt):
        #Euler method for now
        #self.force()
        #self.ax, self.ay = force(self)

        self.wall_collision() #Check for collision

        #self.vx += self.ax * dt / 2
        #self.vy += self.ay * dt / 2

        #print(np.shape(self.r))
        #print(np.shape(self.v))
        #print(np.shape(self.a))

        self.v += self.a * dt / 2

        #self.x += self.vx * dt
        #self.y += self.vy * dt
        self.r += self.v * dt

        #self.vx += self.ax * dt / 2
        #self.vy += self.ay * dt / 2
        self.v += self.a * dt / 2


    def wall_collision(self):
        #Bounce back if hit wall

        span = 2 #Parameter for matching size of particle with wall collision
        elastic = False

        if elastic:
            if np.abs(self.r[0]) >= (self.box_size - span*self.sz):
                self.v[0] = -self.v[0]
            if np.abs(self.r[1]) >= (self.box_size - span*self.sz):
                self.v[1] = -self.v[1]
            if np.abs(self.r[2]) >= (self.box_size - span*self.sz):
                self.v[2] = -self.v[2]
        else:
            damp = 0.95 #Dampening parameter simulating energy loss when hitting wall in non-elastic collision
            damp2 = 0.98 #Smaller damping parameter simulating friction (losing x-vel when hitting y-wall, and vice versa)
            if np.abs(self.r[0]) >= (self.box_size - span*self.sz):
                self.v[0] = -self.v[0] * damp
                
                #self.v[2] = self.v[2] * damp2
                #self.v[1] = self.v[1] * damp2

            if np.abs(self.r[1]) >= (self.box_size - span*self.sz):
                self.v[1] = -self.v[1] * damp
                
                #self.v[0] = self.v[2] * damp2
                #self.v[2] = self.v[2] * damp2

            if np.abs(self.r[2]) >= (self.box_size - span*self.sz):
                self.v[2] = -self.v[2] * damp
                
                #self.v[0] = self.v[0] * damp2
                #self.v[1] = self.v[1] * damp2

def earth_grav(p1):
    #Apply earth gravity
    g = 9.81
    p1.a[2] = -g

def collision(p1, p2):

    def change_velocities(p1, p2):

        m1, m2 = p1.m, p2.m
        M = m1 + m2
        r1, r2 = p1.r, p2.r
        d = np.linalg.norm(r1 - r2)**2

        v1, v2 = p1.v, p2.v

        u1 = v1 - 2*m2 / M * np.dot((v1-v2),(r1-r2)) / d * (r1 - r2)
        u2 = v2 - 2*m1 / M * np.dot((v2-v1),(r2-r1)) / d * (r2 - r1)
        
        p1.v = u1
        p2.v = u2

    if (np.abs(p1.r[0] - p2.r[0]) <= p1.sz) and (np.abs(p1.r[2] - p2.r[2]) <= p1.sz):
        change_velocities(p1,p2)

def init_particles(n,m, sz):
    """Initialize n particles in a box of box_size 10"""
    my_seed = 1
    np.random.seed(my_seed)
    speed_range = 10

    x0 = np.random.uniform(-Particle.box_size, Particle.box_size, n)
    y0 = np.random.uniform(-Particle.box_size, Particle.box_size, n)
    z0 = np.random.uniform(-Particle.box_size, Particle.box_size, n)

    #Starting at non-zero velocity
    vx0 = np.random.uniform(0.1, speed_range, n)
    vy0 = np.random.uniform(0.1, speed_range, n)
    vz0 = np.random.uniform(0.1, speed_range, n)

    #Starting at zero velocity
    #vx0 = [0. for x in range(n)]
    #vy0 = [0. for x in range(n)] 
    #vz0 = [0. for x in range(n)]

    particle_id = [x for x in range(n)] #List of particles IDs

    particle_list = []

    for i in range(n):
        particle_list.append(Particle(x0[i], y0[i], z0[i], vx0[i], vy0[i], vz0[i], 0, 0, 0, m, particle_id[i], sz) ) #Initialize particles

        #Set random positions and velocities of instances in particle_list
        p = particle_list[i] #Current particle
        
        p.r[0], p.r[1], p.r[2] = x0[i], y0[i], z0[i]
        p.v[0], p.v[1], p.v[2] = vx0[i], vy0[i], vz0[i]
    
    return particle_list


def time_sim(n, m, time_steps, dt, sz):
    particles = init_particles(n,m, sz)

    x_pos = np.zeros((time_steps, n)) # (x,n)
    y_pos = np.zeros((time_steps, n)) # (y,n)
    z_pos = np.zeros((time_steps, n)) # (z,n)

    for i in range(n):
        #Set initial positions
        particle = particles[i]
        x_pos[0,i] = particle.r[0]
        y_pos[0,i] = particle.r[1]
        z_pos[0,i] = particle.r[2]


    for t in range(1,time_steps):
        #Loop over time steps

        for j in range(n):
            #Loop over particles
            
            particle = particles[j]
            
            #Save position
            x_pos[t,j] = particle.r[0]
            y_pos[t,j] = particle.r[1]
            z_pos[t,j] = particle.r[2]

            #Particle-particle collision. Doesn't work properly yet.
            part_coll = False
            if part_coll:
                for k in range(n):
                    #Loop over other particles (except itself) and handle collisions
                    if j==k:
                        pass
                    else:
                        collision(particle, particles[k])
            else:
                pass

            earth_grav(particle) #Apply earth gravity

            particle.move(dt)
        
        if t % (time_steps/10) == 0:
            print('Time step {} out of {}  ({}%) done.'.format(t,time_steps, t/time_steps*100))
        
    print('Done')
    return x_pos, y_pos, z_pos

def animate_box(dim_3 = False, save = False):
    """ Animation function. dim_3 = True gives 3D animation, else 2D animation. save_mp4 = True saves animation to file"""

    def get_pos(i):
        #A little backhanded, but I need a update function for animation
        x = x_arr[i]
        y = y_arr[i]
        z = z_arr[i]

        return x,y,z

    # Animation function.
    def animate(i, dim_3, title):
        dx, dy, dz = get_pos(i)
        title.set_text('Particle box, time={:.2f}'.format(i*dt))
        
        if dim_3:
            graph.set_data(dx, dy)
            graph.set_3d_properties(dz)
        else:
            graph.set_data(dx,dz)

        return title, graph


    def plot(dim_3):
        fig = plt.figure()
        wall_p = 30 #Wall parameter for aestethics, so that the particles are not inside the wall

        #Initial values to draw first plot
        x = x_arr[0]
        y = y_arr[0]
        z = z_arr[0]

        if dim_3:
            print('3D animation')
            ax = fig.add_subplot(111, projection='3d')
            graph, = ax.plot(x, y, z, linestyle="", marker="o", color='red')
            ax.set_xlim3d(-box_size - wall_p*sz, box_size + wall_p*sz)
            ax.set_ylim3d(-box_size - wall_p*sz, box_size + wall_p*sz)
            ax.set_zlim3d(-box_size - wall_p*sz, box_size + wall_p*sz)

        else:
            print('2D animation')
            ax = fig.add_subplot(111)
            graph, = ax.plot(x, z, linestyle="", marker="o", color='red')
            ax.set_xlim(-box_size - wall_p*sz, box_size + wall_p*sz)
            ax.set_ylim(-box_size - wall_p*sz, box_size + wall_p*sz)
            #ax.set_zlim(-box_size - wall_p*sz, box_size + wall_p*sz)
            
        title = ax.set_title('Particle box') #Empty title, gets filled in animation


        return fig, ax, title, graph

    fig, ax, title, graph = plot(dim_3)

    frame_num = int(np.floor((steps)))
    anim = animation.FuncAnimation(fig, animate, frame_num, fargs={dim_3, title}, interval=0, blit=True)

    #Saving animation
    if save == 'mp4':
        if dim_3:
            name = 'box3D.mp4'
        else:
            name = 'box2D.mp4'
        print('Saving animation as .mp4')
        anim.save(name, fps=240, extra_args=['-vcodec', 'libx264'])
    
    # Saving as gif with imagemagick does not work for some reason?
    # elif save == 'gif':
    #     if dim_3:
    #         name = 'box3D.gif'
    #     else:
    #         name = 'box2D.gif'
    #     print('Saving animation as .gif')
    #     anim.save(name, writer='imagemagick', fps=60)
    else:
        pass

    plt.show()

def main():
    global box_size, sz, steps, dt, x_arr, y_arr, z_arr #This is shit, I'll fix later
    box_size = 10 #Size of particle box
    dt = 5e-3 #Time step
    time_start = 0
    time_end = 10
    steps = int(np.ceil( (time_end - time_start) / dt)) #Number of time steps

    num_part = 20 #Number of particles
    m = 1 #Mass
    sz = 1e-2 #Size of particle

    x_arr, y_arr, z_arr = time_sim(num_part, m, steps, dt, sz) #Simulating and saving positions

    animate_box(dim_3 = True, save='mp4') #Running animation


if __name__ == "__main__":
    main()