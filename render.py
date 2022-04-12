import numpy as np
from mayavi import mlab

activity = np.load('activity.npz', allow_pickle=True)['arr_0']
input = np.random.randint(0, 2, (28, 28))

fig = mlab.figure(bgcolor=(0, 0, 0))

x, y = input.nonzero()
z = np.zeros_like(x)
#mlab.points3d(x, y, z, mode='cube', scale_factor=0.5, color=(1, 1, 1))

x, y = (input == 0).nonzero()
z = np.zeros_like(x)
#mlab.points3d(x, y, z, mode='cube', scale_factor=0.5, color=(0.2, 0.2, 0.2))

fc1 = np.random.rand(28 * 28, 2000)
fr, to = (fc1 > 0.999).nonzero()
#fr_x, fr_y = np.unravel_index(fr, (28, 28))
#to_x, to_y = np.unravel_index(to, (45, 45))
fr_x, fr_y = np.indices((28, 28))
to_x, to_y = np.indices((45, 45))
fr_x = fr_x.ravel()
fr_y = fr_y.ravel()
to_x = to_x.ravel()
to_y = to_y.ravel()
fr_z = np.zeros_like(fr_x)
to_z = np.ones_like(to_x) + 20

fr_x = fr_x + np.random.rand(len(fr_x)) * 1
fr_y = fr_y + np.random.rand(len(fr_y)) * 1
fr_z = fr_z + np.random.rand(len(fr_z)) * 10
to_x = to_x + np.random.rand(len(to_x)) * 1
to_y = to_y + np.random.rand(len(to_y)) * 1
to_z = to_z + np.random.rand(len(to_z)) * 10

# mlab.points3d(fr_x, fr_y, fr_z, mode='cube', scale_factor=0.5, color=(0.5, 0.5, 0.5))
# mlab.points3d(to_x, to_y, to_z, mode='cube', scale_factor=0.5, color=(0.5, 0.5, 0.5))

# Create the points
x = np.hstack((fr_x, to_x))
y = np.hstack((fr_y, to_y))
z = np.hstack((fr_z, to_z))
s = np.hstack((fc1.mean(axis=1), fc1.mean(axis=0), np.zeros(25)))
src = mlab.pipeline.scalar_scatter(x, y, z, s)
mlab.points3d(x, y, z, s, mode='cube', scale_factor=0.5, scale_mode='none')

# Connect them
src.mlab_source.dataset.lines = np.vstack((fr, to + len(fr_x))).T
#src.mlab_source.dataset.lines = np.array([[0, len(fr_x)], [1, len(fr_x) + 1]])
src.update()

# The stripper filter cleans up connected lines
lines = mlab.pipeline.stripper(src)

# Finally, display the set of lines
mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=0.2)

# And choose a nice view
#mlab.view(33.6, 106, 5.5, [0, 0, .05])
#mlab.roll(125)
#mlab.show()
