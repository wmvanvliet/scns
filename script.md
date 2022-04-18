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
Computational modeling of semantic representation
	Spreading activation model
	Early IAC models
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
	Comparing models of vision to the brain, BrainScore.
	Examples of modern models of written words (mine), spoken words
Comparing models of processes to the brain
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
I will first show you a couple of models of representation and then we'll get into models of processing.

This is an 
