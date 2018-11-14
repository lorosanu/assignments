# Deploy a MNIST classification model on Cloud ML Engine

## Set up the project and credentials on [GCP](https://cloud.google.com/)

* create a new project: mnist (generated id: mnist-222517)
* enable Cloud Machine Learning Engine and Compute Engine APIs
	* APIs & Services > Library
	* search, select and enable
* set up credentials
	* APIs & Services > Credentials 
		* Create credentials > Service account key
		* name: lorosanu
		* role: Project/Owner
	* store the key in the current working directory under auth/auth-key.json

## Docker execution

* set up project id and path for authentication file
	```
	$ PROJECT_ID=mnist-222517
	$ AUTH_FILE=auth/auth-key.json
	```

* train the MNIST models locally
	* shallow model
		```
		$ docker-compose run --rm devel train-local $PROJECT_ID $AUTH_FILE shallow
		```
	* deep model
		```
		$ docker-compose run --rm devel train-local $PROJECT_ID $AUTH_FILE deep
		```

* train the shallow & deep MNIST models on google cloud
	* shallow model
		```
		$ docker-compose run --rm devel train-cloud $PROJECT_ID $AUTH_FILE shallow
		...
		Job [mnist_train_shallow_1] submitted successfully.
		accuracy = 0.9657227, loss = 0.118188664
		```
	* deep model
		```
		$ docker-compose run --rm devel train-cloud $PROJECT_ID $AUTH_FILE deep
		...
		Job [mnist_train_deep_1] submitted successfully.
		accuracy = 0.98583984, loss = 0.04434336
		```

* deploy the MNIST models
	* shallow model
		```
		$ docker-compose run --rm devel deploy-model $PROJECT_ID $AUTH_FILE shallow 1 v1
		```
	* deep model
		```
		$ docker-compose run --rm devel deploy-model $PROJECT_ID $AUTH_FILE deep 1 v2
		```

* run a batch prediction on the deployed MNIST models
	* shallow model
		```
		$ docker-compose run --rm devel predict-batch $PROJECT_ID $AUTH_FILE shallow v1
		```
	* deep model
		```
		$ docker-compose run --rm devel predict-batch $PROJECT_ID $AUTH_FILE v2
		```

* cleaning up
	```
	$ docker-compose run --rm devel clean-up $PROJECT_ID $AUTH_FILE
	```
