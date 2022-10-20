from operator import inv
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from .pf_base import PFLocaliserBase
import math
import rospy

from .util import rotateQuaternion, getHeading
from random import random,gauss,uniform

from time import time

import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


class PFLocaliser(PFLocaliserBase):

    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # ----- Set motion model parameters
        #Number of particles to be plotted initially
        self.number_of_particles = 400
        #Odometric model noises from the pf_base.py
        self.ODOM_ROTATION_NOISE = 0.00001 #odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.000001 #odometry x axis (forward)noise
        self.ODOM_DRIFT_NOISE = 0.00000004 #odometry y axis (side-side) noise
        #particle diffusion parameters to resample when he robot is replaced
        self.diffusion_percentage = 20/100 #rate of particles to diffuse on the entire map


        # ----- Sensor model parameters
        #sensors noise to calculate the poses with the measurement
        self.noise_x = 0.2
        self.noise_y = 0.3
        self.noise_orientation = 120#noise in orientation
        #noise of particles while resampling
        self.resampling_noise_x = 0.1
        self.resampling_noise_y = 0.08
        self.resampling_orientation_noise =15

        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict


    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise
        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.

        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        #initializing the poses of the robot after determining the 2d estimated pose
        Initial_x=initialpose.pose.pose.position.x #position of the robot in X cocordinates
        Initial_y=initialpose.pose.pose.position.y #position of the robot in Y cordinates
        Initial_theta=initialpose.pose.pose.orientation #rotation of the robot with Z axix
        pose_array=PoseArray() # array to return the poses of the robot

        for i in range(self.number_of_particles):
            particle_pose=Pose()
            particle_pose.position.x=Initial_x+gauss(0,1)*self.noise_x
            particle_pose.position.y=Initial_y+gauss(0,1)*self.noise_y


            particle_pose.orientation=rotateQuaternion(Initial_theta, gauss(0,1)*self.noise_orientation*math.pi/180) #determining orientation of the robot with Quaternions from radians
            pose_array.poses.append(particle_pose)

        #returning the particle cloud
        return pose_array




    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.

        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update
         """
        #allocating some weights for the poses of the robot if the robot seems to be having high probability in correct position
        weighted_poses = []
        for pose in self.particlecloud.poses:
            weighted_poses.append(self.sensor_model.get_weight(scan,pose))#weights are determined by the get_Weight() function in sensor_model.py

        sum_of_weights = sum(weighted_poses)
        #normalizing the weights
        weighted_poses = [weight/sum_of_weights for weight in weighted_poses]
        #distributing the weights in a cumilative distribution functional order
        cdf = []
        cdf.append(weighted_poses[0])
        for i in range(1,len(weighted_poses)):
            cdf.append(cdf[i-1]+weighted_poses[i])
        # applying resampling algorithm for the poses
        M=self.number_of_particles-int(self.number_of_particles*self.diffusion_percentage)

        inverse_M =1/M
        resampled_particles = []# store the resampled particles
        U=0
        while U==0:
            U=uniform(0,inverse_M)#creating an uniform distribution
        l=0
        for t in range(M):
            while(U>cdf[l]):
                l+=1
            resampled_particles.append(self.particlecloud.poses[l])
            U+=inverse_M
        updated_array = []
        # Adding noise to the particles
        for i in range(len(resampled_particles)):
            particle_with_noise=Pose()
            particle_with_noise.position.x=(resampled_particles[i].position.x+gauss(0,1)*self.resampling_noise_x)
            particle_with_noise.position.y=(resampled_particles[i].position.y+gauss(0,1)*self.resampling_noise_y)
            particle_with_noise.orientation=rotateQuaternion(resampled_particles[i].orientation, gauss(0,1)*self.resampling_orientation_noise*math.pi/180)
            updated_array.append(particle_with_noise)
        #Adding some particles diffused around the Map
        for i in range(int(self.number_of_particles*self.diffusion_percentage)):
            random_partical_pose=Pose()
            random_partical_pose.position.x=uniform(0,self.occupancy_map.info.width)
            random_partical_pose.position.y=uniform(0,self.occupancy_map.info.height)
            random_partical_pose.orientation.w=1
            random_partical_pose.orientation=rotateQuaternion(random_partical_pose.orientation,uniform(0,1)*self.noise_orientation*math.pi/180)
            updated_array.append(random_partical_pose)
        #updating the particlecloud with updated one
        self.particlecloud.poses=updated_array





    def estimate_pose(self):


        particles = self.particlecloud.poses
        # Set a list to store the sorted particles
        particles_sorted = []
        # For each particle make a 2D list
        # First element: is the index of the particular particle
        # Second element: the weight of that particle
        for i in range(0, len(particles)):
            particle_w = particles[i].orientation.w
            particles_sorted.append([i, particle_w])

        # Sort the 2D array based on their weight (higher ones on top)
        particles_sorted = sorted(particles_sorted, key=lambda x: x[1], reverse=True)

        # Take the first column of the 2D list
        # The order of the indexes of the particles sorted by weight
        indexs = [i[0] for i in particles_sorted]

        # Selecting the top 50% of the particles based one the weight (best particles)
        best_particles = [particles[indexs[i]] for i in range(0, len(indexs)//2)]



        df = pd.DataFrame(columns = ['x_val','y_val', 'orientation','weights'])

        def group_partt(df,best_particles):

            for i,t in enumerate(best_particles):

                pos_x = t.position.x
                pos_y = t.position.y
                w = t.orientation.w

                o1 = t.orientation
                yaw = getHeading(o1)

                df.at[i,'x_val'] = pos_x
                df.at[i,'y_val'] = pos_y
                df.at[i,'orientation']= yaw
                df.at[i,'weights'] = w

            #scaler = StandardScaler()
            #df_scaled = scaler.fit_transform(df)

            kmeans = KMeans(n_clusters = 4, random_state = 2)
            kmeans.fit(df)


            grp = pd.DataFrame()
            grp['data_index'] = df.index.values
            grp['groups'] = kmeans.labels_

            p1 = grp[grp.groups == 1]
            p2 = grp[grp.groups == 2]
            p3 = grp[grp.groups == 3]

            #pa1 =PoseArray()
            #for n in p1:
                #p1. iloc[n]


            return p1,p2,p3

        p1,p2,p3 = group_partt(df,best_particles)


        s1 = []

        s2 = []

        s3 = []
        n1 = 0
        n2 = 0
        n3 = 0
        for i in range(len(p1.index)):
            s1.append(p1.loc[:,['data_index']])

        for i in range(len(p1.index)):
            s2.append(p2.loc[:,['data_index']])

        for i in range(len(p1.index)):
            s3.append(p3.loc[:,['data_index']])


        t1 = []
        t2 = []
        t3 = []

        s11 = np.array(s1)
        s22 = np.array(s2)
        s33 = np.array(s3)

        for i1 in s11[:,0]:
            i = int(i1)
            print(i1,i)
            t1.append(best_particles[i].orientation.w)
            n1=n1+1
        for i2 in s22[:,0]:
            i = int(i2)
            t2.append(best_particles[i].orientation.w)
            n2=n2+1
        for i3 in s33[:,0]:
            i = int(i3)
            t3.append(best_particles[i].orientation.w)
            n3=n3+1




        m1 = sum(t1)/n1
        m2 = sum(t2)/n2
        m3 = sum(t3)/n3
        s = []

        if m1>m2 and m1>m3:
            s = s1
        elif m2>m3 and m2>m1:
            s = s2
        else:
            s = s3

        xp = []
        yp = []
        xo = []
        yo = []
        zo = []
        wo = []

        s = np.array(s)



        for i1 in s[:,0]:
            i = int(i1)
            xp.append(best_particles[i].position.x)
            yp.append(best_particles[i].position.y)
            xo.append(best_particles[i].orientation.x)
            yo.append(best_particles[i].orientation.y)
            zo.append(best_particles[i].orientation.z)
            wo.append(best_particles[i].orientation.w)

        m_xp = sum(xp) / len(xp)
        m_yp = sum(yp) / len(yp)
        m_xo = sum(xo) / len(xo)
        m_yo = sum(yo) / len(yo)
        m_zo = sum(zo) / len(zo)
        m_wo = sum(wo) / len(wo)

        # Putting the estimates pose values into an Pose Object
        est_pose = Pose()

        est_pose.position.x = m_xp
        est_pose.position.y = m_yp
        est_pose.position.z = 0.0
        est_pose.orientation.x = m_xo
        est_pose.orientation.y = m_yo
        est_pose.orientation.z = m_zo
        est_pose.orientation.w = m_wo



        """
        # Particle Coordinates - used for the estimate pose
        xpos = []
        ypos = []

        # Particle Headings - used for estimate pose (orientation)
        xor = []
        yor = []
        zor = []
        wor = []

        # Extracting the coordinates for each best particle
        for particle in best_particles:
            xpos.append(particle.position.x)
            ypos.append(particle.position.y)
            xor.append(particle.orientation.x)
            yor.append(particle.orientation.y)
            zor.append(particle.orientation.z)
            wor.append(particle.orientation.w)

        # Calculating the average pose
        meanxpos = sum(xpos) / len(xpos)
        meanypos = sum(ypos) / len(ypos)
        meanxor = sum(xor) / len(xor)
        meanyor = sum(yor) / len(yor)
        meanzor = sum(zor) / len(zor)
        meanwor = sum(wor) / len(wor)

        # Putting the estimates pose values into an Pose Object
        est_pose = Pose()

        est_pose.position.x = meanxpos
        est_pose.position.y = meanypos
        est_pose.position.z = 0.0
        est_pose.orientation.x = meanxor
        est_pose.orientation.y = meanyor
        est_pose.orientation.z = meanzor
        est_pose.orientation.w = meanwor
        """

        # Return the estimate pose
        return est_pose



    """
        estimated_pose = Pose()

        for p in self.particlecloud.poses:
            estimated_pose.position.x += p.position.x
            estimated_pose.position.y += p.position.y
            estimated_pose.orientation.x += p.orientation.x
            estimated_pose.orientation.y += p.orientation.y
            estimated_pose.orientation.z += p.orientation.z
            estimated_pose.orientation.w += p.orientation.w

        length_of_partical_cloud=len(self.particlecloud.poses)

        estimated_pose.position.x /= length_of_partical_cloud
        estimated_pose.position.y /= length_of_partical_cloud
        estimated_pose.orientation.x /= length_of_partical_cloud
        estimated_pose.orientation.z /= length_of_partical_cloud
        estimated_pose.orientation.y /= length_of_partical_cloud
        estimated_pose.orientation.w /= length_of_partical_cloud

        return estimated_pose
        """
