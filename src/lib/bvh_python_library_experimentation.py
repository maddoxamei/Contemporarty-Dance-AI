from bvh import Bvh

with open('../Data/BVH/Vasso_Miserable_0.bvh') as data:
  mocap = Bvh(data.read())

#Get Mocap Tree
tree = [str(item) for item in mocap.root]
#['HIERARCHY', 'ROOT Hips', 'MOTION', 'Frames: 1068', 'Frame Time: 0.0333333']

#Get ROOT OFFSET
root = next(mocap.root.filter('ROOT'))['OFFSET']
#['3.93885', '96.9818', '-23.2913']

#Get all JOINT names
joints = mocap.get_joints_names() #54
mocap.get_joints_names()[17] #All Names
mocap.joint_offset('Head') #Offset
mocap.joint_channels('Neck') #Channels
mocap.get_joint_channels_index('Spine')
#
mocap.joint_parent_index('Neck') #Parent Index
mocap.joint_parent('Head').name # Parent Name
#mocap.joint_direct_children('Hips') #Direct Children

#Get Frames
mocap.nframes
mocap.frame_time
mocap.frame_joint_channel(22, 'Spine', 'Xrotation')

#SEARCH
[str(node) for node in mocap.search('JOINT', 'LeftShoulder')] #Single Item
[str(node) for node in mocap.search('JOINT')] #All Items



print(len(joints))