pipeline {
    agent any 
    stages {
        stage('Stage 1 - checkout') {
            steps {
                    checkout scm
                }
        },
        stage('Stage 2 - bash script') {
            steps {
                ./process_data.sh
            }
        }
    }
}
