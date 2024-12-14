In order to run the project, run `docker-compose build`, then `docker-compose up` from the root directory of the project. Then navigate to localhost:3000 in your browser to view the user interface.

You must have docker installed to run this app. Docker was used, among other reasons, because it handles the dependency management for us and prevents the user from having to go through multiple steps to get the app up and running. 

The above steps have only been tested on windows. If you have any problems, you can run each individual peice by running `npm run dev` from the `/ui` directory and `flask --app predictor --debug run --port=5000` from the `/predictor` directory. 