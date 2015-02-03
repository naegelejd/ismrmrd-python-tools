"""
Simple tiled image display
"""
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RectangleSelector

def imshow(image_matrix, tile_shape=None, scale=None, titles=[], colorbar=False):
    """ Tiles images and displays them in a window.

    :param image_matrix: a 2D or 3D set of image data
    :param tile_shape: optional shape ``(rows, cols)`` for tiling images
    :param scale: optional ``(min,max)`` values for scaling all images
    :param titles: optional list of titles for each subplot
    """
    assert image_matrix.ndim in [2, 3], "image_matrix must have 2 or 3 dimensions"

    if image_matrix.ndim == 2:
        image_matrix = image_matrix.reshape((1, image_matrix.shape[0], image_matrix.shape[1]))

    if not scale:
        scale = (np.min(image_matrix), np.max(image_matrix))

    if not tile_shape:
        tile_shape = (1, image_matrix.shape[0])
    assert np.prod(tile_shape) >= image_matrix.shape[0],\
            "image tile rows x columns must equal the 3rd dim extent of image_matrix"

    # add empty titles as necessary
    if len(titles) < image_matrix.shape[0]:
        titles.extend(['' for x in range(image_matrix.shape[0] - len(titles))])

    if len(titles) > 0:
        assert len(titles) >= image_matrix.shape[0],\
                "number of titles must equal 3rd dim extent of image_matrix"

    viewer = Viewer(image_matrix, tile_shape, scale, titles, colorbar)
    viewer.show()

class Viewer(object):
    def __init__(self, images, shape, scale, titles, colorbar=False):
        self.fig = plt.figure()
        self.value_text = self.fig.text(.1, .1, "value: 0", bbox=dict(facecolor='white'))
        self.stats_text = self.fig.text(.1, .2, "", bbox=dict(facecolor='white'))
        self.selectors = []
        cols, rows = shape
        for idx, im in enumerate(images):
            ax = self.fig.add_subplot(cols, rows, idx+1)
            ax.set_title(titles[idx])
            ax.set_axis_off()
            imgplot = ax.imshow(im, vmin=scale[0], vmax=scale[1], picker=True)
            self.selectors.append(ROISelector(ax, im, self.update_stats))

            if colorbar is True:
                self.fig.colorbar(imgplot)

        # self.colorbar = CheckButtons(ax, ('Colorbar',), (False,))
        # self.colorbar_active = False
        # self.colorbar.on_clicked(self.toggle_colorbar)
        self.fig.canvas.callbacks.connect('pick_event', self.on_pick)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    def on_pick(self, event):
        if isinstance(event.artist, matplotlib.image.AxesImage):
            x, y = event.mouseevent.xdata, event.mouseevent.ydata
            im = event.artist
            A = im.get_array()
            self.value_text.set_text("value: %.04f" % A[y, x])

    def toggle_colorbar(self, label):
        if self.colorbar_active:
            # disable colorbar
            for ax in self.fig.get_axes():
                for im in ax.get_images():
                    if im.colorbar:
                        im.colorbar.remove()
            self.colorbar_active = False
        else:
            # enable colorbar
            for ax in self.fig.get_axes():
                for im in ax.get_images():
                    self.fig.colorbar(im)
            self.colorbar_active = True

    def update_stats(self, stats):
        lines = []
        for stat in stats:
            lines.append("%s: %0.04f" % (stat, stats[stat]))
        self.stats_text.set_text('\n'.join(lines))
        self.fig.canvas.draw()

    def show(self):
        plt.show()

class ROISelector(object):
    def __init__(self, axes, image, on_update):
        self.image = image
        self.on_update = on_update
        self.rectprops = dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True)
        self.selector = RectangleSelector(axes, self.onselect, rectprops=self.rectprops)

    def onselect(self, eclick, erelease):
        x1, x2 = int(eclick.xdata), int(erelease.xdata)
        y1, y2 = int(eclick.ydata), int(erelease.ydata)
        if x1 != x2 and y1 != y2:
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            subset = self.image[y1:y2, x1:x2]
            self.on_update({'min':subset.min(), 'max':subset.max(), 'mean':subset.mean()})
