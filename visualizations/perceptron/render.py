import numpy as np
from scipy.spatial import distance
from mayavi import mlab
import torch
from tqdm import tqdm
from sklearn.manifold import MDS
import imageio.v2 as iio

from net import Net

model = Net().cpu()
model.load_state_dict(torch.load('mnist.pth', map_location="cpu"))
model.eval()
activity = np.load('activity.npz')

# Distance computations
print('Computing distances...', flush=True)
D_fc1 = np.nan_to_num(distance.squareform(distance.pdist(activity['fc1'].T, metric='correlation')), nan=1)
print('done.')

print('Computing MDS...', flush=True)
fc1_pos = MDS(dissimilarity='precomputed').fit_transform(D_fc1)
print('done.')

fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(3840, 2160))
#fig = mlab.figure(bgcolor=(0.05, 0.05, 0.05))

# Layer units
in_z, in_x = np.indices((28, 28))
#fc1_z, fc1_x = np.indices((32, 32))
#fc2_z, fc2_x = np.indices((32, 32))
#fc3_z, fc3_x = np.indices((32, 32))
fc1_z, fc1_x = fc1_pos.T * 20 + 16
in_x = in_x.ravel() - 12
in_z = in_z.ravel() - 12
in_y = np.zeros_like(in_x)
fc1_x = fc1_x.ravel() - 16
fc1_z = fc1_z.ravel() - 16
fc1_y = np.ones_like(fc1_x) + 10
out_x = np.arange(10)
out_x = out_x.ravel() - 5
out_y = np.ones_like(out_x) + 40
out_z = np.zeros_like(out_x)

#fc1_x = fc1_x + np.random.rand(len(fc1_x)) * 1
#fc1_z = fc1_z + np.random.rand(len(fc1_z)) * 1
fc1_y = fc1_y + np.random.rand(len(fc1_y)) * 10
out_x = out_x * 3

# Connections between layers
fc1 = model.fc1.weight.detach().numpy().T
out = model.fc2.weight.detach().numpy().T
fr_in, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_out = (np.abs(out) > 0.1).nonzero()
fr_fc1 += len(in_x)
to_fc1 += len(in_x)
to_out += len(in_x) + len(fc1_x)

# Create the points
x = np.hstack((in_x, fc1_x, out_x))
y = np.hstack((in_y, fc1_y, out_y))
z = np.hstack((in_z, fc1_z, out_z))

act_input = activity['input'][0]
act_fc1 = activity['fc1'][0]
act_out = activity['fc2'][0]
s = np.hstack((
    act_input.ravel() / act_input.max(),
    act_fc1 / act_fc1.max(),
    act_out / act_out.max(),
))

# Layer activation
acts = mlab.points3d(x, y, -z, s, mode='cube', scale_factor=0.5, scale_mode='none', colormap='gray')

# Connections
src = mlab.pipeline.scalar_scatter(x, y, -z, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_in, fr_fc1)),
    np.hstack((to_fc1, to_out)),
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

# Text
mlab.text3d(x=-14.5, y=40.5, z=-2, text='0')
mlab.text3d(x=-11.5, y=40.5, z=-2, text='1')
mlab.text3d(x=-8.5, y=40.5, z=-2, text='2')
mlab.text3d(x=-5.5, y=40.5, z=-2, text='3')
mlab.text3d(x=-2.5, y=40.5, z=-2, text='4')
mlab.text3d(x=0.5, y=40.5, z=-2, text='5')
mlab.text3d(x=3.5, y=40.5, z=-2, text='6')
mlab.text3d(x=6.5, y=40.5, z=-2, text='7')
mlab.text3d(x=9.5, y=40.5, z=-2, text='8')
mlab.text3d(x=12.5, y=40.5, z=-2, text='9')

mlab.view(azimuth=0, elevation=80, distance=100, focalpoint=[0, 35, 0], reset_roll=False)

# Update the data and view
fps = 60
# movie_writer = iio.get_writer("mnist.mp4", format="FFMPEG", mode="I", fps=fps, codec="hevc_vaapi"
@mlab.animate(delay=1000//fps, ui=False)
def anim():
    for frame in tqdm(list(range(12800))):
        if frame % fps == 0:
            i = frame // fps
            act_input = activity['input'][i]
            act_fc1 = activity['fc1'][i]
            act_fc2 = activity['fc2'][i]
            act_fc3 = activity['fc3'][i]
            act_out = activity['fc4'][i]
            s = np.hstack((
                act_input.ravel() / act_input.max(),
                act_fc1 / act_fc1.max(),
                act_fc2 / act_fc2.max(),
                act_fc3 / act_fc3.max(),
                act_out / act_out.max(),
            ))
            acts.mlab_source.scalars = s
            connections.mlab_source.scalars = s
        mlab.view(azimuth=(frame / 2) % 360, elevation=80, distance=120, focalpoint=[0, 35, 0], reset_roll=False)
        mlab.savefig(f'frames/frame{frame:05d}.png', size=(3840, 2160))
        yield

#anim()
