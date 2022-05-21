# Image-to-Image-Translation-with-Conditional-Adversarial-Networks-Colorization

# Section 1: Background of our project

Image-to-Image Translation

In our project we are taking black and white images and converting them into color images. Networks learn the mapping from input channel (L) to output channels (ab) and try to generate image using latent space that can fool the discriminator.

• We are using as CNN Generator and CNN Discriminator to achieve our results.

Process of converting from black and white to color:

• We are going to be using GAN for our image conversion.

• Converting between 2 different color spaces (RGB – LAB)

LAB and RGB color space:

The 'Lab' contains all possible colors there can be no deterioration in the image quality as all colors translate unaltered.

Lab' has a mix of one channel with no color (L)

• L' channel is a Greyscale.

Two channels with a dual color combination that have no contrast (a+b).

• ‘a’ is the color balance between Green and Magenta.

• ‘b’ is the color balance between Blue and Yellow.

![Picture1](https://user-images.githubusercontent.com/66534109/169672310-c76a58e2-2f0d-455f-982b-d29d425c4559.jpg)
 
# Section 2: Problem setting and goal

Previous Methods:

• Sketch2photo: internet image montage

• A non-local algorithm for image denoising

• Colorful image colorization

Required loss functions and architectures designed specifically for the task at hand.

Goal: A general-purpose solution to image-to-image translation problems.

Our project: use the same architecture and objective for each image-to-image translation task.

• By the use of multiple deep learning nets

• Designed in such a way to solve problems that require lot of human editing/input

• This way there’s less verifying done in each step in the process

# Section 3: Dataset

There were plenty of datasets to train are available, our team choose CIFAR 10 dataset which is a 10,000 images size of 32x32. We just trained our model through all these datasets not having a validation set because we are only trying to output the generated image in the latent space. To test how good the model to be used in the real-world day to day scenarios we tested on some real images and the results are all in the results section. I made no changes to dataset other than normalizing it (range of (-1,1)) and changing to a different color space there may be some information loss when converting between two different color spaces.


# Section 4: Methods

CNN:

The convolutional neural network (CNN/ConvNet) is a type of deep neural network used to evaluate visual images. Convolution is a mathematical operation on two functions that results in a third function that expresses how the form of one is changed by the shape of the other. A grayscale image is nothing more than a matrix of pixel values with a single plane, whereas an RGB image is nothing more than a matrix of pixel values with three planes. Typically, the first layer extracts basic information like horizontal or diagonal edges. This information is passed on to the next layer, which is responsible for detecting more complicated features like corners and combinational edges. As we go deeper into the network, it can recognize even more complex elements like objects, faces, and so on.

The classification layer generates a series of confidence ratings (numbers between 0 and 1) based on the activation map of the final convolution layer, which indicate how likely the image is to belong to a "class."

GANs:

GANs learn a loss that attempts to classify whether the output picture is real or false while also training a generative model to reduce the loss. GANs may be used for a variety of tasks that would normally need quite diverse types of loss functions. If we are dealing with images, this needs to generate images. If we are dealing with speech, it needs to generate audio sequences, and so on. We call this the generator network.

The second component is the discriminator network. It attempts to distinguish fake and real data from each other. Both networks are in competition with each other. The generator network attempts to fool the discriminator network. At that point, the discriminator network adapts to the new fake data. This information, in turn is used to improve the generator network, and so on.

GAN Architecture:

![Picture2](https://user-images.githubusercontent.com/66534109/169672340-ad86b555-ca0b-48e2-a110-5829256a17ad.jpg)

Loss Graphs for GAN:

![Picture3](https://user-images.githubusercontent.com/66534109/169672347-265ba4b7-e555-4ad2-9015-3fd36ee1ce3a.png)
   
cGANs:

A conditional generative model is learned via conditional GANs (cGANs). As a result, cGANs are well-suited to image-to-image translation tasks, in which we condition on an input image and produce a corresponding output image.

![Picture4](https://user-images.githubusercontent.com/66534109/169672369-6dfa4c67-c3ef-4b77-ac58-d45c421213b9.jpg)

• Color distribution matching property of the cGAN

![Picture5](https://user-images.githubusercontent.com/66534109/169672381-6e4dc69d-d96f-4b26-89b6-9c92b8403232.jpg)

• Generator:

![Picture1](https://user-images.githubusercontent.com/66534109/169672443-a5e5759b-224c-4c2a-921f-50f2df9797eb.jpg)

![Pictur1](https://user-images.githubusercontent.com/66534109/169672454-05f60dbd-90d8-418e-aee9-ee24b1323f01.jpg)

• Discriminator:

![Pictu1](https://user-images.githubusercontent.com/66534109/169672459-49e2c3b9-3a7a-4743-b8c3-6a0755542b2f.jpg)

![Pie1](https://user-images.githubusercontent.com/66534109/169672462-0a224865-f63b-4491-8681-12c596b7eb02.jpg)

![Pic1](https://user-images.githubusercontent.com/66534109/169672476-a883c2e6-1816-4f2e-8339-bb53e9fbc04c.jpg)

![Pictur](https://user-images.githubusercontent.com/66534109/169672478-1e6b77d4-6ca2-453e-b45a-f8b60c4c1461.jpg)

Steps for Training - Proposed plan:

● Training the Discriminator
  
  ○ Run a real image through the Discriminator and do a loss function be BCE with 1.
  
  ○ Run a fake image through the Discriminator and do a loss function be BCE with 0.
  
  ○ Add the 2 loss and propagate the error back to account/update for loss in the
    weights in the net.

● Training the Generator
  
  ○ Run a fake image through the Discriminator and get the loss function be BCE with 1
  
  ○ Calculate the pixel-to-pixel loss using the L1 loss of real image and the fake image
  
  ○ Add the 2 loss and propagate the error back to account/update for loss in the weights in     the net.

● Adam Optimizer
  
  ○ Learning rate for Discriminator is lrD = 0.00001
  
  ○ Learning rate for Generator is lrG = 0.0007
  
Experiment Details:

batch_size = 32

lrG = 0.0007

lrD = 0.00001 

Adam(Discreminator.parameters(), lr = lrD) 

Adam(Genarator.parameters(), lr = lrG) 

Genarator_criterion = BCELoss() 

Discreminator_criterion = BCELoss()

L1 = L1Loss()

# Section 5: Experiment Results

The table below shows the results of model on some sample images:

<img width="484" alt="Screen Shot 2022-05-21 at 6 46 37 PM" src="https://user-images.githubusercontent.com/66534109/169672546-65440c96-c91e-46ea-a8bd-2caed0f4a1f2.png">


                                              
Results of Video Conversion:

The table below shows some the loss throughout the training data and this loss is also in the README for further analysis, In the loss below GAN loss Discriminator loss and L1 loss is recorded:


Epoch: 1, batch: 10, LossD: 0.7008737921714783, LossPix2Pix: 0.29060733318 32886, LossG: 0.9965353012084961

Epoch: 2, batch: 10, LossD: 0.6978210210800171, LossPix2Pix: 0.16390667855 739594, LossG: 0.8524675369262695

Epoch: 3, batch: 10, LossD: 0.7038848400115967, LossPix2Pix: 0.19130204617 977142, LossG: 0.8997945189476013

Epoch: 4, batch: 10, LossD: 0.7050642967224121, LossPix2Pix: 0.15198259055 614471, LossG: 0.8469395637512207

Epoch: 5, batch: 10, LossD: 0.7036232948303223, LossPix2Pix: 0.16359166800 9758, LossG: 0.8645942211151123

Epoch: 6, batch: 10, LossD: 0.6971706748008728, LossPix2Pix: 0.15069776773 45276, LossG: 0.8416435122489929

Epoch: 7, batch: 10, LossD: 0.6935681104660034, LossPix2Pix: 0.14407828450 202942, LossG: 0.8385871648788452

Epoch: 8, batch: 10, LossD: 0.6956055760383606, LossPix2Pix: 0.14489594101 905823, LossG: 0.8458768129348755

Epoch: 9, batch: 10, LossD: 0.7043427228927612, LossPix2Pix: 0.13675427436 828613, LossG: 0.8284953832626343

Epoch: 10, batch: 10, LossD: 0.7010186314582825, LossPix2Pix: 0.1362323164 9398804, LossG: 0.829861581325531

Epoch: 20, batch: 10, LossD: 0.708299994468689, LossPix2Pix: 0.14692436158 657074, LossG: 0.8402946591377258

Epoch: 30, batch: 10, LossD: 0.6896252632141113, LossPix2Pix: 0.1530329883 0986023, LossG: 0.8471691608428955

Epoch: 40, batch: 10, LossD: 0.6955165863037109, LossPix2Pix: 0.1310223788 022995, LossG: 0.8350650668144226

Epoch: 50, batch: 10, LossD: 0.691256582736969, LossPix2Pix: 0.12944744527 339935, LossG: 0.8315538167953491

Epoch: 60, batch: 10, LossD: 0.6910091638565063, LossPix2Pix: 0.1356314718 723297, LossG: 0.8446056842803955

Epoch: 70, batch: 10, LossD: 0.6909642219543457, LossPix2Pix: 0.1240091547 369957, LossG: 0.8275965452194214

Epoch: 80, batch: 160, LossD: 0.6951166987419128, LossPix2Pix: 0.126820072 53170013, LossG: 0.8396065831184387
