What is going on during brain activation? 
Computational models
	Mimic what the brain is doing
	At various levels: biological -> macro-scale (we are interested here in macro-scale computations)
	Laboratory for testing different theories of brain function
	Evaluate theories by compare activity in the model to brain activity
	Deep learning revolution has yielded very high-quality tools to build such models.
Modeling representations vs. modeling the process
    Representations are "snapshots in time"
    Process is how to get from one representation to the next
Computational modeling of semantic representations
	Spreading activation model
	Connectionist models
	Feature norms
	Word embedding models (co-occurance, word2vec)
	Sentence embedding models (BERT, GPT-3)	
Comparing representations between model and brain
    Decoding
    RSA
Computational modeling of cognitive processes
	Early IAC models
	Modern models based on deep learning come from machine learning domain, brain inspired, but not explicitly meant as brain models
	CNN revolution: HMAX and AlexNet winning ImagetNet competition by crushing the competition
	Examples of modern models of written words (mine), spoken words
Comparing models of processes to the brain
	BrainScore.
    My study
    Feed through unseen types of stimuli, explore boundary conditions.
The loop: observation -> theory -> prediction -> refine theory


Hello everyone, my name is Marijn van Vliet and I will tell you about the role of computer models in the study of language in the brain.
As Riitta Salmelin has told you during her overview, whenever we show a simulus to a volunteer in the scanner that contains language in some form, we can observe a series of bursts of activity in the brain.
The question I would like to focus on now, is what the brain is actually doing during these bursts of activity.
What computational processed are being performed as the stimulus hits our senses, gets translated into electrical impulses, and travels through the brain?
To be honest, we still know very little about this, but one thing we can tell you is that it is a complicated process involving hundreds of millions of neurons in different regions of the brain.
So we're probably going to need a bit more than just pen and paper to work this out.
What we need are computer models that try to mimic the computational processes.

 - They are given the same stimuli as the volunteer in the scanner
 - They perform the same task as the volunteer in the scanner, obtaining the same behavioral result
 - They do the task in such a way that the activity within the model mimics (parts of) the brain activity we observe

Such models provide a useful laboratory for exploring new theories and making predictions from them that can be tested against the brain activity we recorded during our experiments.
The better the theory, the better the model, the more closely the model can predict the brain activity.

How closely does the model need to simulate the brain?
Neuroscience is being performed at many different scales, all the way from understanding the protein structures that make up the cells, to the neurons themselves, to connections between brain regions and the human body as a whole.
When studying the language functions of the brain, we are generally concerned somewhere at this level: large networks of neurons that together perform some computation, like detecting a letter, or retrieving the meaning of a word.
 
At this point, I want to start showing you some computational models that are currently being used to study language.
But in order to appreciate what these models are trying to accomplish, we must first make a distinction between models of *representation* and models of *processing*.
Models of representation aim to organize data in such a way as it might be organized in the brain.
They are only concerned with simulating the final organization, not how this organization comes about.
In contrast, models of *processing* aim to model how the brain transforms data from one representation into another.

Let's take a look at some models of representation.

One of the deep questions of language in the brain is the representation of abstract semantics.
A word acts as a trigger for the activation of rich semantic concept with a deeper meaning than just the word.
How are these concepts and their meaning represented in the brain?
We don't know.
But if we want to find out, we need to start somewhere.
Start with a models that are maybe too simplistic, but that we can actually build, so we can learn from them, and then try to improve them.

A core theme in many, if not all, modern models of semantics is that the meaning of a concept is closely tied to the relationships it has to other concepts.
What is a "dog"?
It is an "animal" with "fur" that "barks" and "wags" its "tail".

This inspires us to model semantics in the brain as a vast network of related concepts.
The model you are looking at has been constructed with the help of tens of thousands of volunteers that participated in the Small World of Words project, led by Simon De Deyne.
The volunteers have been presented with a cue word and were asked to write down the first three words that came to mind.
We can then draw connections between words based on their responses.
In such a model, whenever a concept is activated, connected concepts are co-activated, which in turn activate even more concepts, bringing about a rich semantic representation.

This network view of semantics has been useful for modeling a phenomenon in the brain known as "semantic priming".
You have already seen semantic priming at work during Riitta Salmelin's overview.
When a word is not expected, based on the words that preceeded it, there is a larger burst of activity in the temporal cortex.
But semantic priming does not require complete sentences.
It also works when showing a word pair, just two words that are either related or unrelated.
You are looking here as some data recorded with EEG instead of MEG, so the N400 looks different, but the priming effect is the same.
So, reading even a single word already influences how your brain will process related words in the future.
If we want to model this effect using our network, we can activate the first word in the model and simulate the amount of co-activation of the second word.
The amount of co-activation that we simulated is a good indicator for the strength of the priming effect we observe in the brain reponse.

In the Small World of Words model, connections can be drawn between any two words.
We could however try to organize the connections a bit more.
Here is a network model proposed by David Rumelhart and Peter Todd in 1993.
We have nouns on the left.
Each noun maps to a combination of "attributes", here shown on the right.
Which types of attributes are activated is modulated by these relationship types shown at the bottom.
For example, if we activate the noun Canary and the relationship type "CAN", we activate the attributes "Can Grow" "Can Move" "Can Fly" and "Can Sing".
To achieve this, there are additional nodes in this network that do not represent words or concepts, but merely function as connection hubs for spreading activation.
Such nodes are called "hidden nodes", we will see more of them in others models.

A group of nodes is called a "layer", we will also see more of those in other models.
Now, this model is more powerful than our initial network model in two ways.
First, we can model different kind of relationships between concepts, which is always useful.
But second, this layer on the right, the "attributes" layer, holds a pretty efficient representation of the meaning of a concept.
Each concept is represented as a unique combination of attributes.
A SPUR is a TREE, TALL, GREEN, has BARK and ROOTS, and so on.
GRASS is also GREEN and has ROOTS, but is a PLANT rather than a TREE and is not TALL, and so on.
Even with a limited set of attributes, we can represent a great number of different concepts.

Can you think of a way to simulate the semantic priming effect in this model?

Here is one way:
We could have attributes stay active for a while.
When a new word is presented that shares many of the same attributes as the preceding words, those attributes will have been pre-activated, hence the word will be processed faster.

Representing concepts as a collection of attributes is pretty efficient.
But there is even a more efficient way.
And that is this "hidden layer" here.
All activity passes through this layer on its way to the "attribute layer", so all the information needed to identify a concept must already be present at this stage.
And there are far fewer nodes in this hidden layer than in the attribute layer.
The nodes in this layer form what we call a "semantic embedding space" and this is currently one of the most popular ways to represent the meaning of concepts.
To get an intuition for what is going on this layer, its best to leave this model behind for now and talk about another model called "word2vec".

The "word2vec" model was developed by a research team at Google, led by Tomas Mikolov, and published in 2003.
The way it works is a follows.
We read in a large amount of text. Books, news articles, subtitles from television programs, anything we can get our hands on.
We take all the unique words we've encountered in the text, usually a couple of million, and assign them a random position in a space which we call the "semantic embedding space".
Here I am drawing a 3-dimensional embedding space so its easy to visualize, but usually you take a space with many dimensions.
Now we go over the text again and every time two words occur together in the same sentence, we move those words a little closer together in the embedding space.
And every once in a while, we stop and move words that have never occured together a little further apart.
We keep doing this until we read all of the text we had collected, and for good measure, start from the beginning of the text again and again until the positions of the words in the embedding space no longer change much.
At that point, we say that the model has "converged".
The resulting positions of the words in the embedding space have very interesting properties.

First of all, we have achieved a very efficient representation, as we can identify millions of individual words just by their coordinates in the embedding space.
Again, I've used only 3 dimensions in this example, so each word is represented by only 3 numbers, but in a real model you would use about 300 dimensions.

A second interesting property of this model is that words that are in close proximity are frequently used either in the same sentence or are frequently used in the same contexts.
Words that are close in the embedding space have a similar meaning.

A third interesting property of this model is that directions in the embedding space have meaning too.
This is something you will explore on your own during an exercise I will introduce at the end of this lecture.

Now that you are familiar with the concept of an embedding space, I can explain what is going on inside the hidden layer of the previous model.
Each node in this layer represents one of the dimensions of an embedding space.
Together, the activation across these nodes represents the location of a concept in the embedding space.
There!

Lets now look at how embedding spaces, such as this word2vec space, are used as models of representation for semantics in the brain.
I'd like to illustrate this with some fMRI data that was collected by Sasa Kivisaari et al., published in 2019.
During this study, the participants were asked to solve little riddles as a way to make them focus on different concepts.
For example, to get them to focus on the concept of a "banana", they would be told to think about something that was yellow, sweet and eaten by monkeys.
So here is the brain activity as the participant was focusing in a banana, as recorded through fMRI.
And here is the brain activity for a helicopter.
And a cow.
On the left, you see a semantic embedding space, projected down to 2 dimensions for easy visualization.
You can see the original 300-dimensional coordinate of the word as a bar graph below the brain.
Now here's the interesting thing:
The closer words are in the embedding space, the more similar the corresponding brain activity patterns.
In a way, the brain activity we record acts as an embedding space of its own.
If you treak every voxel as a dimension in embedding space, you get an embedding space with a crazy high dimensionality, and the pattern of activity across the voxels represents a location in this embedding space.
So now we have two embedding spaces, the one we created with the word2vec algorithm using a large amount of text, and the one defined by the brain activity.
The former acts as a computational model for the latter.
By creating a linear mapping from the word2vec embedding space to the brain activity embedding space, we can predict the brain activity pattern of a word, based on its location in word2vec space.

In 2016, Huth et al. published a study that explores the possibilities of such a mapping in detail.
They collected brain activation patterns in response to many words by having participants listen to stories in an fMRI scanner.
Using machine learning, they created a mapping between a word2vec embedding space and brain activity patterns.
The model could predict the activity in some areas of the brain better than others.
Notably, activity in the areas associated with semantic processing was predicted best.
The proceeded to examine for each voxel in the brain, the words that activated that voxel the strongest.
And they found that each voxel has a preference for words from within a certain region of the word2vec embedding space.
We already saw that words that are close together in embedding space are similar in meaning.
So what the authors of the study did was assign category labels to different regions of the embedding space.
You can see them at the bottom left.
They gave each category a different color.
Now, a preference for words from a certain region becomes preference for a category.
Voxels can thus be assigned a preference for a certain category, yielding this pretty semantic atlas of which part of the brain have a preference for which categories.
This voxel is most active for words related to visual features.
And this one for words concerning relationships.
And this one to numbers.
Admittedly, this is streching the mapping between the computational model and the brain activity a bit far and I'm not certain how accurate this truly is.
But it does demonstrate how far you can take computational models of representation.































