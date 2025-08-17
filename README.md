# AI 
It refers to the ability of computer systems to perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.
<br>
Note : ML is a subset of AI and DL is a subset of ML. And AI means providing 
"Intelligence" to machine. 
<br>
Today's AI is subset of human intelligence (Pattern recognition) because human 
intelligence is made up multiple things like Pattern recognition, Imagination, emotional 
intelligence. 
<br>
The first wave of AI is called "Symbolic AI" in which knowledge-based system and expert system was made. 
<br>
knowledge-based System: This a set of programs which contains multiple if-else checks to give the correct output. 
<br>
Expert System: This is also a set of programs (or we can say decision-making system) which are made by taking the knowledge of any expert.
Example :  Apps in which we can play chess with an expert. 
<br>
Disadvantage of Expert System : It is applicable on closely-related problem. 
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
Important use case of ML . 
<br>
Data mining is to create a prediction model by applying the machine learning algorithm. 
<br>
This prediction model gives us some pattern. 
<br>
Example : Email spam classifying. 

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
When we have a dataset which contains input and output but the type of output is numeric. 
<br>
Example : Suppose we have a dataset which has three columns where first 2 columns are input and third column is 
output. First column is "IQ", second column is "CGPA" and third column is "Placement Package". 
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
Example : Suppose we have a dataset which has two columns where first column is "IQ",second column is "CGPA" 
then we can plot a graph using these inputs and get some information.
For Example: Identify the category of student who has maximum IQ and Max CGPA.
<br>

(II) Dimensionality Reduction : 
<br>
In some problem of Supervised Learning, we  have a lot of input columns in dataset then 
Supervised learning algorithm is slow down. In such problem Dimensionality 
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
The main function of the Anomaly Detection is to detect the unexpected behavior in the system.
<br> 
Example : Loan Approval Anomaly. 
<br>
When we apply for a loan to any bank then there are types of profile verification is done. 
first verification is done by using ML algorithm. ML Engineers compare our profile with some previous defaulters
who did not payback the money. If there is strong correlation between them and us then our loan will be 
rejected. If there is less correlation then a manual verification is done by bank employee and loan will be 
sanctioned.
<br>
(IV) Association Rule Learning : 
<br>
A type of Unsupervised Learning where we fetch the useful information from dataset 
(data mining) and draw conclusion. 
<br>
Its main goal is to find the relationship or patterns between variables in the large dataset.
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
positive points and on every incorrect actions, it gets negative points. In this way agent 
learns from data and update its policy. 
<br>
Example : self driving car, Go game solver etc. 
<br>
For self-driving-car, the environment is road. 



# Types of ML : 
On the basis of how ML models behaves in production environment, It is of two types :
<br>
<br>

<b>1.Batch ML / Offline Learning : </b> 
<br>
It is the conventional approach where a model is trained using the entire available dataset at once. The training process happens offline, meaning the model does not learn or update in real time. Once trained, the model is deployed and used for prediction on new data without further training. This is its main disadvantage.
<br>
Note : We can pull down the model from the server and add new data to it and train model again and then 
deployed on the server again. This cycle can be repeated after some period of time like after 24 hrs. 
<br>
It is used where there is no concept drift. like : Image-Classification etc. 
<br>
No Concept Drift: It means the condition for predicting the output based on input will never changed in future. 
<br>
Tools used : scikit-learn, tensorflow, Pytorch,etc.
<br>
Disadvantage of Batch ML : 
<br>
Loss of data : When data becomes so large like Big data which cannot be processed by tools used to 
train Ml model. 
<br>
Hardware Limitation : suppose when we do not have instant connectivity with our ML model. In this case,
we cannot able to instantly update and deployed ML to the server and it will reach to users. 
<br>
Availability : suppose when we are working on social media application which shows the content on the basis of 
user's interest. suppose an unexpected event occurs like "demonetization" then everyone shows their interest 
in demonetization but our ML model will upadte data after some period of time (it can be 24hrs or else). 

<br>
<br>

<b>2. Online ML / Online Learning : </b>
<br>
Online Machine Learning is a type of machine learning where the model is trained incrementally, which means it learns from one data point at a time or mini batches as they arrive — instead of training on the full dataset at once (like in batch learning).
<br>
As these data chunk is very small so that models can be trained on the server using these data. 
<br>
Examples : Chat bots like siri etc.
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
Learning Rate : 
<br>
It means how frequently we update our model. 
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
1. Tricky to use : There are few libraries which we can use to train model. For the 
enterprized application, where the data is so large, it is difficult to process these data 
to train model using these libraries.  
<br>
2. Risky : If the behavior of our model depends on incoming request and suppose if anyone hacks the incoming 
request then it is risky for our model because it can make our model biased. 
<br>
To prevent model from this risk, there should be monitoring system which continuously monitors the incoming 
request and if there is anomaly in the incoming request then monintoring system can reject that request. 
<br>
Another if any risk occurs then we must have the facility of rollback to go to the previous state. 
<br>



# Types of ML : 
On the basis of how ML model learns : 
<br>
1. Instance based Learning : 
<br>
It is a type of lazy learning method in machine learning where the model memorizes the data and makes predictions by comparing new examples to stored ones.
<br>
Example : Using KNN.
<br>
<br>
2. Model based Learning : 
<br>
Model-Based Learning is a machine learning approach where the algorithm builds a general model from training data before making predictions. In simple words, it learn patterns and relationships in the data during training and then use that model for predictions.
<br>
Example : Using Linear Regression, Logestic Regression etc.


# Chalanges in ML : 
<b>1. Data collection: </b> 
<br>
In company level ML project we get the data by "making api calls" and "web scraping". 
<br>
Fetching the data by these two methods can cause many problems. 
<br>
<b>2. Insufficient Data / Labelled Data  :</b>
If we have Insufficient data then we may face many difficulties to train a model. 
<br>
Getting the labelled data is also very costly. 
<br>
Note : When we have enough data then it does not matter which algorithm we are using. 
And this is called Unreasonable Effectiveness of data. 
<br>
But in present gathering enough data in proper format is very tricky. So algorithm matters. 
<br>
<b>3. Non Representative data : </b>
<br>
If we have not complete dataset then our model will not give more accuracy. This is also 
called Sampling Noise. 
<br>
Sometimes data can be biased which is also called sampling bias. for Example : We coducted a 
survey to know who will win T-20 World-Cup in which we took response from all countries 
who will participate in the World-Cup. But in this survery, most people vote for "India"
because these people supports india which lives in India as well as in other country.
<br>
<b>4. Poor Quality Data : </b>
<br>
Data may contains outliers, missing value, data in improper format and other things which need to be clean before training a model on thes data. 
<br>
Note : It is said that "ML Engineers spend 60% of time on cleaning the data while training a model."
<br>
<b>5. Irrelevant Features : </b>
<br>
If we have Irrelevant features in our dataset then our model will not perform good. 
<br>
Example : If we have a dataset which is used to predict who will participate in the 
marathon. And the data set contains some field like Age, Height, Weight and Location. 
Then in this case Location is Irrelevant features because it does not affect the result.
<br>
<b>6. Overfitting : </b>
Overfitting in machine learning occurs when our model memorizes data, not understanding the pattern. And this type of model will not work on new data. 
<br>
<b>7. Underfitting : </b> 
It is just opposite of overfitting. Underfitting in machine learning occurs when a model
is too simple to capture the underlying patterns in the data. As a result, it performs 
poorly on both the training data and unseen data (test set).
<br>
Example : Always trying to fit a straight line to data shaped like a curve or spiral.
<br>
<b>8. Software Integration : </b>
ML model always integrates with some software to help users. 
<br>
It is not easy to integrate the model with software because of different plateform. 
like : linux, windows etc. 
<br>
<b>9. Offline Learning / Deployment</b>
<br>
In Offline Learning if we need to update model the we need to pull down the model from server and add new data 
to it and train model again and then deployed on the server again which is very costly. 
<br>
<b>10. Cost Involved : </b>
Server cost of Ml model is generally very high. 

# Application of ML : 
<b>Application of ML in B2C (Business to Customer)   : </b>
<br>
1. Youtube uses ML to suggest videos. 
<br>
2. Facebook/Instagram uses ML to give friend recommendation. 
<br>
3. Chatbots. 
<br>
<br>
<b>Application of ML in B2B (Business to Business) : </b>
<br>
It helps Businesses to grow and earn more profit. 
<br>
<b>1. Retail - Amazon/Big Bazaar : </b>
<br>
Example : (I) Companies like Amazon, flipkart etc. offers Sale on festivals in which they reduces the prices of 
product and increase the stock of product. But they do not increase the stock of every product. They provide 
last 2 or 3 years customer's purchase related data to their ML Engineers and these engineers gives insights 
from these data to company by applying ML algorithm on these data.  
<br>
(II) When we purchase any product from super-market then they ask our mobile No to make bills. The main reason
behind it is that they create customer purchase related profile by using their mobile no. They want to understand our buying pattern. Like if anyone buys healthy products then it means that person is 
health-oriented he would prefer to go GYM. In this case these super-market can sell these data to GYM owner and earn profit. 
<br>
(III) In super market, how the things are placed so that company can get maximum profit.
It is very difficult to place the things at correct positions. In such case we can use 
Association Rule Learning. We can analyze customer bills and find some hidden patterns. 
like : People purchase egg along with milk so there is strong association between milk and 
egg. so we need to place egg with milk.
<br>
<br>
<b>2. Banking and Finance Sector : </b>
<br>
Example : (I) When we apply for a loan to any bank then there are types of profile verification is done. 
first verification is done by using ML algorithm. ML Engineers compare our profile with some previous defaulters
who did not payback the money. If there is strong correlation between them and us then our loan will be 
rejected. If there is less correlation then a manual verification is done by bank employee and loan will be 
sanctioned. 
<br>
Ml is widely used in stock market and trading and other finance sectors. 
<br>
<br>
<b>3. Transport- OLA : </b>
<br>
Example: (I) There is very high differece in OLA taxi's fares in evening and other times. It is because
there are some offices point where a lot of employee works. But these places are generally at far places. 
But in evening time a lot of people book OLA taxi's. But the taxi driver don't want to go to these far places. 
But OLA offers cab drivers to pay double and that's why OLA increases the fare. These places are find out by
using ML.
<br>
ML is also used in google map. 
<br>
<br>
<b>4. Manufactoring - Tesla: </b>
<br>
In Tesla company, automobiles are made by robots. Tesla cars are booked six months before it actually
manufactured. So the company works by following some schedule and suppose one day a robot whose job is to place 
the engine in the car is malfunction then on that day no cars will be made and it is big loss for that company
because they are following schedule. To prevent from these type of risk they fixed "IOT" device in
robots which continuously moniters their pressure,rpm etc. We know faults occurs gradually in any machine. 
And if any anomalous behaviour detected in any robot then those robots will be fixed by engineers. This is 
called predictive mentainance.
<br>
<br>
<b>5. Consumer Internet- Twitter : </b>
<br>
Most of the social media plateform uses the sentimental analyses to earn money. 
<br>

# ML Development Life cycle (MLDLC): 
It is set of guidelines which we should follow while developing the ML based software product. 
<br>
<b>1. Frame the problem : </b> 
<br>
We need to answer some questions before starting :
<br>
What is the problem exactly?
<br>
Who is our customer ?
<br>
What is the cost for developing such product ?
<br>
How many manpowers are needed to develop such product ?
<br>
How end product looks like ?
<br>
which ML model is needed to implement whether supervised or Unsupervised ?
<br>
Whether ML model will run in offline or online mode ? 
<br>
Which type of algorithm will be used ?
<br>
Where the data comes from ?
<br>
<br>
<b>2. Gethering Data : </b>
<br>
In company level ML project we get the data from various resource : 
<br>
From CSV files,
<br>
By "making api calls" and "web scraping",
<br>
Sometimes data are stored in company's database but we cannot directily run our model on that databases 
becauese if any fault occurs in database then websites/apps can be down. In this case, We need to create
data warehouse. From this data warehouse we fetch the data. 
<br>
Sometimes data are stored in clusters (Big data) and in this case, we need to fetch data from there. 
<br>
<br>
<b>3. Data preprocessing : </b>
<br>
Intially data does not clean. So we need to preprocessing these data. 
<br>
Following steps are done at this stage  :
<br>
Remove duplicates 
<br>
Remove missing values,
<br>
Remove outliers,
<br>
Scale up/down the column's value. 
<br>
<br>
<b>4. Exploratory Data Analysis : </b>
<br>
Here we analyze the data and extract the relationship between different variables. 
<br>
Here, we plot graph for visulization,
<br>
Here, we do univariate analysis which means independent analysis on each column - mean, standard deviation, 
which curve is followed. We also do bivariate analysis which means analysis on two different column. 
Sometimes we also do multi-variats analysis. 
<br>
Outliers detections
<br>
Balance the Imabalance the data. Imabalance data means there is unequal amount of data in two columns like if 
we do classification problem like dog and cat classification.
<br>
At this stage, we have good understanding about our data. 
<br>
<br>
<b>5. Feature Engineering and Selection : </b>
<br>
Suppose we are working on house dataset which contains 1000 input columns. 
like two columns are - "No_Of_Rooms" and "No_Of_Bathrooms" then we create a column "Square_fit_area" using 
these two columns and then removes those two columns. This is called Feature Extraction. This is done because 
in bunch of input, there are some inputs on which result never depends and In such we also feature selection 
in which few inputs are selected to train a model. 
<br>
<br>
<b>6. Model Training, Evaluation and Selection : </b>
<br>
Here we apply different ML algorithms on our model and then we evaluate model using different type of matrix like 
MSE(mean-squared-error) in case of linear Regression and this evaluation helps us to choose the best model. 
And then we do model selection in which we tune the best model algorithm parameters (parameters means setting)
so that it performs better. This is also called Hyper-Parameter-Tuning. 
<br>
A term called "Ensemble Learning" in which we combines multiple ML algorithm and make a powerful algorithm 
which increase the performance of our model. 
<br>
<br>
<b>7. Model Deployment : </b>
<br>
Here we need to integrate the model with some web app or mobile apps so that it serves the users.
<br>
Here we generate a binary file from model and then we create api and when user send the correct input to 
api endpoint then api sends the input to that binary file and then this file predicts and then it send predicted
data to api and then api send response to the user. 
<br>
Then we deploy this websites or mobile apps on server.
<br>
<br>
<b>8. Testing : </b>
<br>
At this stage, we perform Beta testing (testing performed by trusted customers).
<br>
Here, we also perform "A/B Testing" which tells whether our model works good or not and if not then 
we repeat previous steps. 
<br>
<br>
<b>9. Optimize : </b>
<br>
If we found that our model works good in "A/B Testing" then we move ahead and optimize several things : 
<br>
We took backup of our model and data. 
<br>
We setup rollback service. 
<br>
We setup loadbalancing. 
<br>
We decide how frequently we will retrain our model.

# Tensors : 
Tensor is a data structure which is mostly used in ML and DL. 
<br>
It is mostly used to store numbers but sometime it is used to store characters and strings. 
<br>
<b>0D Tensor/Scaler : </b>
<br>
np.array(3) :  is called scalers or 0D Tensors. 
<br>
<b>1D Tensor/Vector/1D array : </b>
<br>
np.array([1,2,3,4]) :  is called 1-D tensor and vector.
<br>
Note: Dimensions is equal to no of axes and no of axes is equal to rank. 
And shape of tensor is the order of matrix or array. 
And size of tensor is No of elements in a matrix.
<br>
Here, dimension of this tensor is 1 and it'shape is (1,4) and it's size is 4. 
<br>
Note : Here the dimension of this vector is 4. Dimension of vector depends on no of element in a matrix.
because this vector can be represented as [1,0,0,0], [0,2,0,0], [0,0,3,0],[0,0,0,4].
<br>
<b>2D Tensor/Matrices : </b>
<br>
np.array([[1,2,3],[4,5,6]]) : is called 2-D Tensor or Matrix. 
<br>
<b>ND Tensor : </b>
<br>
nd array is called ND Tensor. 
<br>



# How to Frame a Machine Learning Problem : 
<b>step 1 : Converion of Business problem to ML problem. </b>
<br>
First of all we need to convert the business problem into mathematical problem. 
<br>
Example : A Business problem of an OTT plateform "Netflix" is : Increase revenue. 
<br>
Revenue can be increased by three ways : 
<br>
(I) Increase new customer. And It is very difficult to increase new customer.
<br>
(II) Demand for high charge from existing customer. And it is wrong.
<br>
(III) Decrease the no of users who are going to left the platform. This can be right approach. 
<br>
So we choose third way and try to decrease the churn rate. 
<br>
Churn rate is the annual percentage rate at which customers stop subscribing to a service. 
<br>
Like current churn rate is 4 and we try to decrease to 3.75. Now this is a mathematical problem. 
<br>
<b>step 2. Identify the type of problem </b>
<br>
In this step, we have to see big pictures like
<br>
Prediction: Identify the users who are going to left the platform ? 
<br>
Now we can say it is a classification problem of supervised learning where we need to find out whether user are 
going to leave the platform or not.  
<br>
What will the end product ? or what we can do to stop the users to leave the plateform ?
<br>
Immediate plan can be to give discount to those users who are going to leave the platform. 
<br>
But here the problem is if we give same percent of discount to all users who are going to leave the platform 
then it can be a great loss for that company. There may be some users whose probability to leave the plateform
is less or high or very high. 
<br>
Now we understood that this is a Regression problem not classification problem in which we need to identify the 
probability to leave platform for those users who are going to leave the platform. 
<br>
Our long term plan can be to identify the problem of users that why user is going to leave the platform ?
<br>
This problem can be high subscription price, internet issues, UI navigation for very old people etc. 
<br>
<b>step 3.</b>
<br>
Current Solution 
<br>
In this step, We need to communicate with CTO to know whether is there any current solution.
<br>
In this problem the current solution can be a team who is already working on a model which calculates the 
churn-rate. so we can start from there. 
<br>
<b>Step 4: Getting Data </b> 
<br>
Here, we need data to make prediction model which gives information about users who are going to leave the 
platform. 
<br>
Here, we need to identify the data which are required : 
<br>
(I) watch time 
<br>
(II) Search but did not find.
<br>
(III) Content left in the middle.
<br>
(IV) Clicked on recommendation (order of recommendation)
<br>
So in this step, we need data engineers who will fetch these data from application database(OLTP) by creating
warehouse. 
<br>
<b>step 5. Metrics to measure : </b>
<br>
In this step, we need to define the metrices whether we are going into right direction or not. 
<br>
So here, we can check the differece between the current churn-rate and previous churn-rate before discount and 
suppose if the differece is 0.25 then it means we are going into right direction
<br>
We also check whether the users who are going to left the plateform has left or other users. 
<br>
<b>step 6. Online Vs Batch :</b>
<br>
In this step, we have to decide whether we train the model offline or online. 
<br>
In this problem, we would prefer to go with online because there is case of concept drift or condition is 
volatile. 
<br>
<b>step 7 : Check Assumptions </b>
<br>
In this step, we need to check multiple asssumptions. 
<br>
One of the assumption can be : whether this model will be applicable on all country or we should make 
geographically based model. 

# Data Gathering : 
1. CSV Files from multiple platform like Kaggle. 
<br>
Google_colab_link : https://colab.research.google.com/drive/1pKaSxPYDbH0KnLTLwtUXNlhETNoTSwxe?usp=sharing
<br>
2. JSON/SQL data 
<br>
Google_colab_link : https://colab.research.google.com/drive/1h1KUHH66M0e5w8k2Eyjx-UFMNW_UXJnN?usp=sharing
<br>
3. Fetch data from any api. 
<br>
Google_colab_link : https://colab.research.google.com/drive/1rZ9UkYvNp3wpcl1CBtqmD0qJdpQwiSlv?usp=sharing
<br>
4. Get data by WebScraping. 
<br>
Google_colab_link: https://colab.research.google.com/drive/1dVkRa2SYnNcEIj2EcjiXn14volnBUNsm?usp=sharing


# Understanding Data : 
This step is a combination of Data preprocessing and EDA.
<br>
To understand data, first of all we have to ask some basic questions like : 
<br>
Google_colab_link : https://colab.research.google.com/drive/1eTlsVa7kE0MlDnjzMt-hBNdAsIQkW6QA?usp=sharing
<br>
Univariate Analysis (a part of EDA) : 
<br>
Google_colab_link : https://colab.research.google.com/drive/1ydt1TJpqWoZQtTKRKbgOg2lZwJwixO7Z?usp=sharing
