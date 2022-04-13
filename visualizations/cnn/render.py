import numpy as np
from mayavi import mlab
import torch
from tqdm import tqdm

from net import Net

model = Net()
model.load_state_dict(torch.load('mnist.pth'))
model.eval()
activity = np.load('activity.npz')
act_input = activity['input'][0][0]
act_conv1 = activity['conv1'][0]
act_conv2 = activity['conv2'][0]
act_fc1 = activity['fc1'][0]
act_out = activity['output'][0]

fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(1920, 1080))

img_input = mlab.imshow(act_input, colormap='gray', interpolate=False, extent=[-27, 27, -27, 27, 1, 1])
img_input.actor.position = [0, 0, 0]
img_input.actor.force_opaque = True

# Conv1
for row in range(6):
    for col in range(6):
        img_conv1 = mlab.imshow(act_conv1[row * 6 + col], colormap='gray', interpolate=False)
        img_conv1.actor.position = [(row - 2.5) * 26, (col - 2.5) * 26, 50]
        img_conv1.actor.force_opaque = True

# Conv2
for row in range(8):
    for col in range(8):
        img_conv2 = mlab.imshow(act_conv2[row * 8 + col], colormap='gray', interpolate=False)
        img_conv2.actor.position = [(row - 3.5) * 12, (col - 3.5) * 12, 100]
        img_conv2.actor.force_opaque = True

# Create the points
conv2_x = list()
conv2_y = list()
for row in range(8):
    for col in range(8):
        x, y = np.indices((12, 12))
        x = x.ravel() + (row - 4) * 12
        y = y.ravel() + (col - 4) * 12
        conv2_x.append(x)
        conv2_y.append(y)
conv2_x = np.hstack(conv2_x)
conv2_y = np.hstack(conv2_y)
conv2_z = np.ones_like(conv2_x) * 100

fc1_x, fc1_y = np.indices((12, 12))
fc1_x = fc1_x.ravel() * 2 - 12
fc1_y = fc1_y.ravel() * 2 - 12
fc1_z = np.ones_like(fc1_x) * 150 + np.random.rand(*fc1_x.shape) * 10

out_x = np.arange(10)
out_x = (out_x.ravel() - 5) * 3
out_y = np.zeros_like(out_x)
out_z = np.ones_like(out_y) + 200

x = np.hstack([conv2_x, fc1_x, out_x])
y = np.hstack([conv2_y, fc1_y, out_y])
z = np.hstack([conv2_z, fc1_z, out_z])
s = np.hstack([
    act_conv2.ravel() / np.max(act_conv2),
    act_fc1 / np.max(act_fc1),
    1 - (act_out / np.max(act_out)),
])
acts = mlab.points3d(x[len(conv2_x):], y[len(conv2_x):], z[len(conv2_x):], s[len(conv2_x):], mode='cube', scale_factor=1, scale_mode='none', colormap='gray')

# Connections between the layers
fc1 = model.fc1.weight.detach().numpy().T
out = model.fc2.weight.detach().numpy().T
fr_conv2, to_fc1 = (np.abs(fc1) > 0.08).nonzero()
fr_fc1, to_out = (np.abs(out) > 0.2).nonzero()
to_fc1 += len(conv2_x)
fr_fc1 += len(conv2_x)
to_out += len(conv2_x) + len(fc1_x)
c = np.vstack((
    np.hstack((fr_conv2, fr_fc1)),
    np.hstack((to_fc1, to_out)),
)).T

src = mlab.pipeline.scalar_scatter(x, y, z, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_conv2, fr_fc1)),
    np.hstack((to_fc1, to_out)),
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

# Update the data and view
@mlab.animate(delay=83, ui=True)
def anim():
    for frame in tqdm(list(range(1600))):
        if frame % 16 == 0:
            i = frame // 16
            act_input = activity['input'][i]
            act_conv1 = activity['conv1'][i]
            act_conv2 = activity['conv2'][i]
            act_fc1 = activity['fc1'][i]
            act_out = activity['output'][i]
            s = np.hstack((
                act_conv2.ravel() / act_conv2.max(),
                act_fc1 / act_fc1.max(),
                1 - (act_out / act_out.max())
            ))
            acts.mlab_source.scalars = s
            connections.mlab_source.scalars = s
        #mlab.view(azimuth=(frame / 2) % 360, elevation=80, distance=120, focalpoint=[0, 35, 0], reset_roll=False)
        #mlab.savefig(f'/l/vanvlm1/scns/frame{frame:04d}.png')
        yield

#anim()
