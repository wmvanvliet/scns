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

In the Small World of Words model, connections can be drawn between any two words.
We could however try to organize the connections a bit more.
Here is a network model proposed by David Rumelhart and Peter Todd in 1993.
We have nouns on the left.
Each noun maps to a combination of "attributes", here shown on the right.
Which types of attributes are activated is modulated by these relationship types shown at the bottom.
For example, if we activate the noun Canary and the relationship type "CAN", we activate the attributes "Can Grow" "Can Move" "Can Fly" and "Can Sing".
To achieve this, there are additional nodes in this network that do not represent words or concepts, but merely function as connection hubs for spreading activation.
Such nodes are called "hidden nodes".

One way to study semantics in the brain is through the phenomenon of semantic priming.
You have already seen semantic priming at work during Riitta Salmelin's overview.
When a word is not expected, based on the words that preceeded it, there is a larger burst of activity in the temporal cortex.
This semantic priming effect does not require complete sentences.
It also works when showing a word pair, two words that are either related or unrelated.
Reading even a single word already influences how you will process related words in the future.
Going back to the EEG responses to word pairs...
If we activate the first word in the model and measure the amount of co-activation of the second word, we obtain a good indicator for the strength of the priming effect we observe in the EEG reponse.
