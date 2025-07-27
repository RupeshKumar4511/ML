# AI 
It refers to the ability of computer systems to perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.
<br>
Note : ML is a subset of AI and DL is a subset of ML. And AI means providing "Intelligence" to machine. 
<br>
Today's AI is subset of human intelligence (Pattern recognition) because human intelligence is made up multiple 
things like Pattern recognition, Imagination, emotional intelligence. 
<br>
The first wave of AI is called "Symbolic AI" in which knowledge-based system and expert system was made. 
<br>
knowledge-based System: This a set of programs which contains multiple if-else checks to give the correct output. 
<br>
Expert System: This is also a set of programs (or we can say decision-making system) which are made by taking the knowledge of any expert.
Example :  Apps in which we can play chess with an expert. 
<br>
Disadvantage of Expert System : It is applicable of closely-related problem. 
<br>
Machine Learning solves the problem of expert system. 


# ML 
Machine Learning is field of computer science which uses statistical technique to give the 
computer system ability to learn with "data" without being explicitly programmed. 
<br>
In simple words, ML means "learning from data" and it does not require explicit programming. 
<br>
In ML, we gives data to system and then system analyze the data and finds patterns. 
<br>
ML works good for small data.   
<br>
Example : Checking the email is spam or not. 


# Data anlysis : 
Analizing the data by plotting graphs. 

# Data Mining : 
Important use case of ml . 
<br>
Data mining is to create a prediction model by applying the machine learning algorithm. 
<br>
This prediction model gives us some pattern. 
<br>
Example : Email span classifying. 

# Disadvantage of ML : 
1. We need to specify the features of input that we give to our system. 
<br>
2. Performance of ML model stablizes at a point with increase in the amount of data given to system. 
<br>
These problems are solved by the Deep Learning.  

# DL :
Deep learning is a subfield of machine learning that utilizes artificial neural networks with multiple layers to analyze data and learn complex patterns.
<br>
DL is inspired by biology(neuron). But it is just a mathematical model. 
<br>
In Deep Learning, we don't need to specify the features of input. Here System automatically creates the features.
<br>
Performance of DL model always increases with increase in amount of data given to the system.
<br>
DL works good for large data.  

# Types of ML : 
On the basis of amount of supervision required for a machine learning algorithm to get trained,
there are four different types of ML:
<br>
<b>1. Supervised Learning : </b>
<br>
A type of ML where we have a dataset which contains input and output and on the basis of 
input and output and we need to make the prediction on the new input by identifying the 
relationship between input and output. 
<br>
Most of the time, we works on supervised machine learning. 
<br>
It is further divided into two types :
<br>
(I) Regression : 
<br>
When we have a dataset which contains input and output but the type of output is numerical. 
<br>
Example : Suppose we have a dataset which has three columns where first 2 columns are input and third column is output. First column is "IQ", second column is "CGPA" and third 
column is "Placement Package". 
<br>
(II) Classfication : 
<br>
When we have a dataset which contains input and output but the type of output is categorical. example: Gender,Nationality etc. 
<br>
Example : Suppose we have a dataset which has three columns where first 2 columns are input and third column is output in which first column is "IQ", second column is "CGPA" and third column is "Placement" (in terms of "YES/NO"). 
<br>
<br>
<b>2. Unsupervised Learning : </b>
<br>
A type of ML where we have dataset that contains only input and we need to get some 
useful information by using different technique. 
<br>
It is further divided into four types(techniques):
<br>
(I) Clustering :
<br>
In Clustering we make N-dimensional clustors from the dataset.
<br>
Its main goal is to group similar data points together based on their features.  
<br>
Example : Suppose we have a dataset which has two columns where first column is "IQ",second column is "CGPA" then we can plot a graph using these inputs and get some 
information. For Example: Identify the category of student who has maximum IQ and Max CGPA.
<br>

(II) Dimensionality Reduction : 
<br>
In some problem of Supervised Learning, we  have a lot of input columns in dataset then 
Supervised learning algorithm will be slow down. In such problem Dimensionality 
Reduction remove the unnecassary columns.
<br>
Example : Suppose we are working on house dataset which contains 1000 input columns.
like two columns are - "No_Of_Rooms" and "No_Of_Bathrooms" then Dimensionality Reduction
makes a column "Square_fit_area" using these two columns and then removes those two 
columns. This is called Feature Extraction. This is done because in bunch of input, there 
are some inputs on which result never depends. 
<br>
It also helps in visulization. There are certain Clustering problems where we have thousand
of axes which is not possible to draw, in such a case Dimensionality Reduction reduces 
these dimensions to 2-3 dimensions. 
<br>

(III) Anomaly Detection : 
<br>
The main function of the Anomaly Detection is to detect the unexpected behavior in the
system.
<br> 
Example : Loan Approval Anomaly.
<br>
(IV) Association Rule Learning : 
<br>
A type of Unsupervised Learning where we fetch the useful information from dataset 
(data mining) and draw conclusion. 
<br>
Its main goal is to find the relationship or patterns between variables in the 
large dataset.
<br>
Example: In super market, how the things are placed so that company can get maximum profit.
It is very difficult to place the things at correct positions. In such case we can use 
Association Rule Learning. We can analyze customer bills and find some hidden patterns. 
like : People purchase egg along with milk so there is strong association between milk and 
egg. so we need to place egg with milk.


<br>
<br>
<b>3. Semisupervised Learning : </b>
<br>
It is partially supervised and partially Unsupervised learning. 
<br>
In Semisupervised learning, we don't need to label(output) of each input. We labels few 
inputs and all other will be automatically labeled with help of Semisupervised learning.
<br>
This is found because labeling the data is very costly.  
<br>
Example : In Google Photos, when we named a folder myselfie and put our selfie into it then Google photos put all those images to that folder which has the same face as in our first selfie due to Semisupervised learning. 
<br>
<br>
<b>4. Reinforcement Learning : </b>
<br>
In Reinforcement learning we have no data. 
<br>
In Reinforcement learning, there is an agent and agent has some policy(Rule book) and 
agents needs to live in an environment. On every correct action of agent, it gets some
positive points and every incorrect actions, it gets negative points. In this way agent 
learns from data and update its policy. 
<br>
Example : self driving car, Go game solver etc. 
<br>
For self-driving-car, the environment is road. 


<br>
<br>
<br>

On the basis of how ML models behaves in production environment, ML is of two types :
<br>
<br>

<b>1.Batch Ml / Offline Learning : </b> 
<br>
It is the conventional approach where a model is trained using the entire available dataset at once. The training process happens offline, meaning the model does not learn or update in real time. Once trained, the model is deployed and used for inference on new data without further training. This is its main disadvantage.
<br>
Note : Company can pull down the previous data and add new data to it and train model and then deployed on 
the server again. This cycle can be repeated after some time like after 24 hrs. 
<br>
Disadvantage of Batch ML : 
<br>
Loss of data : When data becomes so large like Big data which cannot be processed by tools used to 
train Ml model. 
<br>
Hardware Limitation : suppose when we do not have instant connectivity with our ML model. In this case,
we cannot able to instantly update and deployed to the server and it will reach to users. 
<br>
Availability : suppose when we are working social media application which shows the content on the basis of 
user's interest. suppose an unexpected event occurs like "demonetization" then everyone shows their interest 
in demonetization but our ML model will upadte data after some period of time (it can be 24 hrs or else). 

<br>
<br>

<b>2. Online ML / Online Learning : </b>
<br>
Online Machine Learning is a type of machine learning where the model is trained incrementally, which means it learns from one data point at a time or mini batches as they arrive — instead of training on the full dataset at once (like in batch learning).
<br>
As these data chunk is very small so that models can be trained on the server using these data. 
<br>
Examples : Chat bots like siri etc
<br>
<br>

<b>When to use Online ML : </b>
<br>
1. where there is a concept drift : suppose we have trained a model to solve a particular problem and then
nature of problem changes continuously then we use Online ML. for Example : stock exchange etc. 
<br>
2. Cost Effective : As we work with small data chunk. 
<br>
3. Faster Solution : As models trains on small data chunk so it is faster. 
<br>
<br>
How to implement Online Learning : 
<br>
1. Using Python River Library 
<br>
2. VOWPAL WABBIT python Library : It mainly works for Reinforcement learning but it also allows to
do online ML. 
<br>
<br>
Learning Rate : It means how frequently we update our model. 
<br>
<br>
Out of core Learning : 
<br>
When the data is so large that cannot be processed then with the help of Online Learning technique, we can 
divide the large data into small chunks and then we train model using small data chunk offline and then we
deployed model to server. This concept is called Out of core Learning. 
<br>
<br>
<b>Disadvantage of Online ML : </b>
<br>
1. Tricky to use : There are few libraries which we can use to train model. For the enterprized application,
where the data is so large, it is difficult to process these data to train model using these libraries.  
<br>
2. Risky : If the behavior of our model depends on incoming request and suppose if anyone hackes the incoming 
request then it is risky for our model because it can make our model biased. 
<br>
To prevent model from this risk, there should be monitoring system which continuously monitors the incoming 
request and if there is anomaly in the incoming request then monintoring system can reject that request. 
<br>
Another if any risk occurs then we must have the facility of rollback to go to the previous state. 


