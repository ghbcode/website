---
layout: default_sidebar
title: GHBCode's Site
description: Notes on Comp Sci, Math, Data Analysis, Python and Misc
---

## How to use Flask for a RESTful app

The purpose of this code is to do the following using REST api endpoints:
1. query the [plaid connect api](https://plaid.com/docs/legacy/api/#data-overview) and get transactions from a user 

curl -X POST https://tartan.plaid.com/connect \
   -d client_id=test_id \
   -d secret=test_secret \
   -d username=plaid_test \
   -d password=plaid_good \
   -d type=wells
   
2. return a count of the transactions that are above a threshold amount.


## The code

```
#!/usr/bin/python3
from flask import Flask
from flask_restful import Resource, Api

'''
This was a test given to me by Digit on 20180816
The purpose is to do the following using rest api endpoints:
1. query the plaid connect api and get transactions from a user

https://plaid.com/docs/legacy/api/#data-overview

curl -X POST https://tartan.plaid.com/connect \
   -d client_id=test_id \
   -d secret=test_secret \
   -d username=plaid_test \
   -d password=plaid_good \
   -d type=wells
2. return a count of the transactions that are above a threshold amount.

--------------------How to kill this server: lsof -ti :8015 | xargs kill
'''
class Outlier(Resource):
    def get(self, clientid, threshold):
        ''' Return a count of the transactions whose amount exceeds threshold amount.
        This function is dependent on the Client.post() call
        '''
        client = Client()
        transactions = client.post(clientid, threshold)
        return {'count': transactions}

class Client(Resource):
    def post(self, clientid, threshold):
        '''
        Gets transaction data from plaid and returns the number of occurences above the threshold

        param clientid:
        param threshold:
        '''
        #print(clientid, threshold)
        import requests
        url = 'https://tartan.plaid.com/connect?client_id={0}&secret=test_secret&username=plaid_test&password=plaid_good&&type=wells'.format(clientid)

        res = requests.post(url)

        if res.status_code == 401:
            issue = "Wrong username"
            print(issue)
            return issue
        elif res.status_code == 404:
            issue = "Wrong URI"
            print(issue)
            return issue
        else:
            try:
                transactions = res.json()
                # count number of transactions above threshold
                l = list()
                for trans in transactions['transactions']:
                    if (trans['amount'] > threshold):
                        l.append(trans['amount'])

                return len(l)
            except:
                issue = "Problem with JSON data"
                print(issue)
                return issue

app = Flask(__name__)
api = Api(app)
api.add_resource(Outlier, '/Outlier/<clientid>/<int:threshold>/')

if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port="8015")
    except:
        print("Could not start. Most likely socket is in use. Kill process by: lsof -ti :8015 | xargs kill")
```


## Testing the code

Put the code above in a file (server.py for example) and run it. It will let you know where it is running, for example 127.0.0.1:8015. So if you pull up a web page with the following URI http://127.0.0.1:8015/Outlier/test_id/150/
you should get the result 

    {"count": 4}


## Putting the code in a Docker container 

To avoid any issues with clashing requirement versions I will place this code it its own Docker container. Admittedly, this could very well run on its own virtualenv. I picked Docker since I'm using it more and more nowadays though I admit it is overkill. 

- Create the file Dockerfile
This is where the Docker image is created, updated and run. 
Make sure that the EXPOSE command points to the port where server.py choses to run in.

```
FROM python:3-onbuild
EXPOSE 8015
CMD ["python", "./server.py"]
```

- Create the requirements.txt file
This will be used to install the required packages into the Docker image.

```
flask
flask_restful
requests
```

- Create the Docker image

    docker build -t ghbcode/flasktest:latest .

The build command will download the python 3 image which is about 690MB in size as well as update packages via the requirements.txt file. Once it is done you may run the image as follows:

    docker run -p 8015:8015 --rm ghbcode/flasktest:latest

You should be able to access this service via http://x.x.x.x:8015/Outlier/test_id/150/ where x.x.x.x is the IP of the machine(host) running the Docker image. 

The *docker ps* command on the host will tell you if the container is running and the PORT. In my case it reads 
    
    8015->8015/tcp

### Resources
* Following the topic of encapsulation, often times you will create multiple Docker containers that will work together to create an application. Think for example creating a Docker container for the Database end and another for the code. The following link has a very good article and the food truck example shows two Docker containers working in unison to provide functionality: [Docker for Beginners](https://docker-curriculum.com/)
* This link has very good post install settings to consider when you are using Linux: (https://docs.docker.com/install/linux/linux-postinstall/)[https://docs.docker.com/install/linux/linux-postinstall/]
