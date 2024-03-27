# Segmentation_GUI
The labeling GUI of the segmentation used to help the training of a Neuron network to recognize and segment the neurons of C-elegans 

## Essential Package (Using conda)

- numpy
- matplotlib
- h5py
- tkinter
- PIL(pillow)
- skimage
- pyclesperanto_prototype
- shapely

It is highly recommended to use conda virtual environment to install all the essential pacakes:

To create a new conda environment, follow these steps:

1. Open a terminal or command prompt.
2. Run the following command to create a new conda environment with a specified Python version (for example, Python 3.8):

`conda create --name <my_conda_env> python=3.8`

Replace `<my_conda_env>` with your desired environment name. Change 3.8 to the Python version you want to use.

3. Activate the newly-created conda environment:

• On Linux or macOS:

`conda activate my_conda_env`

• On Windows:

`activate my_conda_env`

After activating the environment, you can install packages using conda or pip, and then running the python script to initate the GUI.

For the installation of the pacakges, you can run this follwoing command using pip3:

`pip3 install numpy matplotlib h5py scikit-image pyclesperanto-prototype pillow tkinter shapely`

## The guideline to use the labeling GUI
### Basic element
Upon opening the GUI, there will be two buttons, a slider on top, and two windows displaying pictures at the bottom.

The two buttons, `Load File` and `Save File`, are self-explanatory. The slider controls the layer of the 3D slices in the input files.

The picture window on the left is where you can locate the working region. You can click or drag using the mouse to create a selection of the working area, which will be shown in the right window after selection.

After choosing the working region, you can use the middle scroll of the mouse or zoom in and out with two fingers on the right picture panel to control the zoom of the working region. This makes selecting neurons easier.

After confirming the zoom level of the working region, one can use the mouse to draw a closed circle to label the neuron. Once the circle is closed, the marked pixels will be automatically labeled with a color.

### Working flow
After creating a mask of the pixel of a neuron on a certain layer, one needs to use the `shift` key to store it. After storing it to the data output, one must move to other layers containing the same neurons, in order to label them with the same color.

After labeling the same neuron within all the possible layers and saving them, one can type the `n` key to start labeling another neuron with a different color.

After the labeling work is done, one need to use the button `Save File` to store all the existing labeling information. One can check-point their work by using `Load File` button.

**Notice**: Once you type n to use other colors, it will be impossible to go back to the previous color. Therefore, it is essential to label all the layers in which the neuron appears before moving on to a new color. A possible remedy to the problem is to cancel all the previous labeling of that neuron using `Ctrl+z` on every layers involves. The detail usage of the short cut key can be seen in the next section.

### Short cut keys
- `Ctrl+z`: Cancel all the previous selection within the working regions.
- `shift`: Store the area
- `n`: Start with a new color
