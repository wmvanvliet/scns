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
img_input.actor.orientation = [0, 90, 90]
img_input.actor.force_opaque = True

# Conv1
img_conv1 = list()
for row in range(6):
    for col in range(6):
        img = mlab.imshow(act_conv1[row * 6 + col], colormap='gray', interpolate=False)
        img.actor.position = [(row - 2.5) * 26, 50, (col - 2.5) * 26]
        img.actor.orientation = [0, 90, 90]
        img.actor.force_opaque = True
        img_conv1.append(img)

# Conv2
img_conv2 = list()
for row in range(8):
    for col in range(8):
        img = mlab.imshow(act_conv2[row * 8 + col], colormap='gray', interpolate=False)
        img.actor.position = [(row - 3.5) * 12, 100, (col - 3.5) * 12]
        img.actor.orientation = [0, 90, 90]
        img.actor.force_opaque = True
        img_conv2.append(img)

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
acts = mlab.points3d(x[len(act_conv2.ravel()):], z[len(act_conv2.ravel()):], y[len(act_conv2.ravel()):], s[len(act_conv2.ravel()):], mode='cube', scale_factor=1, scale_mode='none', colormap='gray')

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

src = mlab.pipeline.scalar_scatter(x, z, y, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_conv2, fr_fc1)),
    np.hstack((to_fc1, to_out)),
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

# Text
mlab.text3d(x=-14.5, y=200.5, z=-2, text='0')
mlab.text3d(x=-11.5, y=200.5, z=-2, text='1')
mlab.text3d(x=-8.5, y=200.5, z=-2, text='2')
mlab.text3d(x=-5.5, y=200.5, z=-2, text='3')
mlab.text3d(x=-2.5, y=200.5, z=-2, text='4')
mlab.text3d(x=0.5, y=200.5, z=-2, text='5')
mlab.text3d(x=3.5, y=200.5, z=-2, text='6')
mlab.text3d(x=6.5, y=200.5, z=-2, text='7')
mlab.text3d(x=9.5, y=200.5, z=-2, text='8')
mlab.text3d(x=12.5, y=200.5, z=-2, text='9')

mlab.view(0, 90, 400, [0, 100, 0])

# Update the data and view
@mlab.animate(delay=83, ui=True)
def anim():
    for frame in tqdm(list(range(1600))):
        if frame % 16 == 0:
            i = frame // 16
            img_input.mlab_source.scalars = activity['input'][i][0]
            for img, a in zip(img_conv1, activity['conv1'][i]):
                img.mlab_source.scalars = a
            for img, a in zip(img_conv2, activity['conv2'][i]):
                img.mlab_source.scalars = a
            act_fc1 = activity['fc1'][i]
            act_out = activity['output'][i]
            s = np.hstack((
                act_conv2.ravel() / act_conv2.max(),
                act_fc1 / act_fc1.max(),
                1 - (act_out / act_out.max())
            ))
            acts.mlab_source.scalars = s[len(act_conv2.ravel()):]
            connections.mlab_source.scalars = s
        mlab.view(azimuth=(frame / 2) % 360, elevation=80, distance=300, focalpoint=[0, 100, 0], reset_roll=False)
        mlab.savefig(f'/l/vanvlm1/scns/frame{frame:04d}.png')
        yield

anim()
