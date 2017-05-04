# CS152 Final Project: Multi-Adversarial Autoencoders <small>By Jonathan Chang and Zachary Friedlander</small>

## Background

In recent years, Generative Adversarial Networks (GANs) have gained popularity
as a method for generating samples that resemble some class of inputs (e.g.
handwritten digits or human faces). The basic GAN architecture consists of two
networks: a generator, which creates the samples, and a discriminator, which
tries to distinguish between real samples and generated samples. The discriminator
is trained to make as few mistakes as possible, while the generator is trained
to fool the discriminator. In this way, the generator learns to create outputs
that actually resemble the inputs.

The success of GANs has led to a number of spinoff architectures based on the
adversarial concept. One such architecture is the [Adversarial Autoencoder](https://arxiv.org/pdf/1511.05644.pdf).
In the Adversarial Autoencoder, the generator is actually an autoencoder
(a type of network which learns to create lower-dimensional encodings of its
inputs). The discriminator is trained to distinguish between the autoencoder's
encodings and samples drawn from a prior distribution (e.g. a Gaussian). Once
again, the discriminator is trained to minimize mistakes, while the generator
has the dual task of learning good encodings while _also_ fooling the
discriminator. The result is that the autoencoder learns to fit its encoding
to the prior distribution. This result is useful because it means that 
samples drawn from the prior distribution can be run through the decoder
part of the autoencoder to produce meaningful outputs.

Another spinoff of GANs is the [Generative Multi-Adversarial Network](https://openreview.net/pdf?id=Byk-VI9eg).
This architecture extends GANs by adding multiple discriminators, rather than just
having one. The training occurs in much the same way as before; each discriminator
is trained to distinguish between real and generated samples, while the generator
can be trained to fool all the discriminators or only the best discriminator (which
might change between iterations). This has been empircally shown to make the
generator converge faster. The theoretical basis for this result is the
machine learning concept of _ensemble learning_, which involves combining the
outputs of multiple independent classifiers to produce a single better classifier.
As such, we believe the multi-adversary concept could be helpful in other 
architectures, including the Adversarial Autoencoder.

## Problem Statement

Investigate the effects of adding multiple discriminators to the Adversarial
Autoencoder framework.

## Experimental Design and Approach

For the remainder of this writeup, we refer to our modified Adversarial Autoencoder
as the Multi-Adversarial Autoencoder.

We implemented a Multi-Adversarial Autoencoder in Python using the Keras library.
We also used a third party extension to Keras, called keras-adversarial. This
extension adds high-level functions for designing adversarial architectures,
and supplies a one-line function for training. We started with code for
a basic Adversarial Autoencoder, which can be found [here](https://github.com/bstriner/keras-adversarial/blob/master/examples/example_aae.py).
We then modified the code to use 5 discriminators, rather than just one. As in
the case of the Generative Multi-Adversarial Network, training of the Multi-Adversarial
Autoencoder works as follows: each discriminator is trained the same way as the
discriminator in a standard Adversarial Autoencoder, while the generator is
trained to fool all 5 discriminators (while also trying to autoencode well).

As previously mentioned, ensemble methods require _independent_ classifiers.
This implies that for the multi-adversarial approach to work, the discriminators
need to be different from each other in some way, so that their outputs do
not end up being the same or substantially similar. We came up with two ideas
for how to make the discriminators different from each other:

- Give each discriminator only a subset of the encoding, rather than seeing the
  entire encoding. For example, rather than trying to distingish the entire
  encoding from an entire sample from the prior, we might have a discriminator
  try to distingish the first 50 elements of the encoding from the first 50
  elements of the sample.
- Vary the learning parameters (e.g. learning rate) for each discriminator.

We decided to try both of these approaches, to compare their impact on performance.
Our experimental procedure is as follows:

1. Run the basic adversarial autoencoder on a dataset for N epochs to gain a baseline for comparisons.
2. Run a multi-adversarial autoencoder using the subset approach for N epochs and compare the results.
3. Run a multi-adversarial autoencoder using the varied parameters approach for N epochs and compare the results.

We wanted to avoid common datasets like MNIST and CIFAR-10, simply because they
have been used so often and there are already models that do very well on them.
Instead, for our dataset, we decided to use Pokemon sprites. These sprites cover
Pokemon from all of the Pokemon games, and are official artwork with a consistent
art style. There are slightly over 800 sprites. To minimize memory usage and
training time, all images were scaled down to 32x32 pixels.

## Results and Conclusions

As mentioned previously, our baseline for comparisons is the standard adversarial
autoencoder. The outputs of the standard adversarial autoencoder after N epochs
of training on our Pokemon dataset are shown below:

INSERT FIGURE HERE

The outputs are definitely fuzzy and not very well-defined. However, this is
somewhat to be expected since autoencoders are a lossy compression method, and
the original artwork is highly detailed. Despite the fuzziness, we can see that
the overall shapes that get generated do seem to resemble Pokemon, at least at
the outline level, and the colors also look similar to what is found in the
original dataset. While these outputs are far from perfect, our goal is comparing
the results of the multi-adversarial autoencoder, not necessarily creating perfect
generated results, so these outputs serve as a suitable baseline.

Our next experiment was with the subset-based multi-adversary approach. For our
implementation, we decided to give each of the 5 adversaries 1/5 of the encoding.
The outputs of this multi-adversarial autoencoder after N epochs of training
on our Pokemon dataset are shown below:

INSERT FIGURE HERE

We find that the generative ability of this network is heavily diminished. The
generated outputs are essentially just shapeless blobs with dull colors. We do
note, however, that the autoencoding seems somewhat improved, with more
recognizable features in the autoencoded samples.

Finally, our last experiment was with the varied parameters approach. For our
implementation, we decided to vary the learning rate and the loss weight for
each discriminator (the loss weight is a value specifying how much each
discriminator's loss contributes to the overall loss). The outputs of this
multi-adversarial autoencoder after N epochs of training are shown below:

INSERT FIGURE HERE

Qualitatively, performance seems fairly similar to the standard adversarial
autoencoder. The generated outputs still look vaguely Pokemon-shaped with
interesting coloring.

From these experiments, we draw the following conclusions:

- Giving each discriminator a subset of the encoding hurts the model's
  generative ability. We hypothesize that this is because slicing up the encoding
  in this way causes the discriminators to fail to fit the encoding to the prior
  distribution. As a result, the model acts as a mere autoencoder, with little
  to no correlation between the prior distribution and the learned encodings.
  This would explain why the generated samples look like formless blobs, as
  samples from the prior distribution would not correspond to anything meaningful
  and thus would not result in meaningful outputs when passed to the decoder.
  This might also explain why the autoencoding seems improved - since the model
  is not fitting the encoding to the prior, it is simply learning the best
  encoding it can.
- Varying the parameters of each discriminator seems to be a more promising
  approach, in that it does not visibly hurt the model's generative ability.
  However, we also did not see any significant improvement in performance either.
  It might be that we need to vary different parameters, or use different ranges
  of values for the parameters we are varying. This may be a good avenue for
  future exploration.

In addition, we also believe the following approaches might be good candidates
for future work on this subject:

- Modifying the subset approach to allow overlapping subsets, rather than strictly
  slicing up the encoding. If the subsets are made large enough, this could avoid
  the problem of failing to fit the encoding to the prior distribution.
- Having the generator only try to fool the best performing discriminator (as
  shown in the original Generative Multi-Adversarial Networks paper) rather than
  trying to fool all the discriminators as we are currently doing. Unfortunately,
  this approach is not possible using the keras-adversarial library, since it
  provides a "black box" training function that cannot be modified in user code.
  Thus, investigating this approach would have required modifying the library
  itself, or rewriting the code in a different neural network library like TensorFlow, which we
  did not have time to do.

## References

Ishan Durugkar, Ian Gemp, and Sridhar Mahadevan. Generative Multi-Adversarial Networks. ICLR 2017.

Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial Autoencoders. arXiv preprint arXiv:1511.05644v2, 2016.

## Code Directory

All our code for this project can be found at the following GitHub repository:
[https://github.com/jpwchang/CS152-Multi-AAE](https://github.com/jpwchang/CS152-Multi-AAE)

## Class Presentations

Below is a link to our project presentation: [https://docs.google.com/a/g.hmc.edu/presentation/d/11mYIJHM_7i1UL27ItKhmGxZtVb2fa_wtHQTkK6EqnNI/edit?usp=sharing](https://docs.google.com/a/g.hmc.edu/presentation/d/11mYIJHM_7i1UL27ItKhmGxZtVb2fa_wtHQTkK6EqnNI/edit?usp=sharing)
