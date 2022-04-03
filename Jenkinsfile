pipeline {
  	agent {
		docker { image 'mikolajpokrywka/ium:0.0.0' }
	}
    parameters {
        string (
            defaultValue: 'mikolajpokrywka',
            description: 'Kaggle username',
            name: 'KAGGLE_USERNAME',
            trim: false
        )
        password(
            defaultValue: '',
            description: 'Kaggle token',
            name: 'KAGGLE_KEY'
        )
        string (
            defaultValue: '17000',
            description: 'cut data',
            name: 'CUTOFF',
            trim: false
        )
    }
    stages {
        stage('checkout: Check out from version control') {
            steps { 
                    checkout([$class: 'GitSCM', branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('bash script') {
            steps {
                withEnv(["KAGGLE_USERNAME=${params.KAGGLE_USERNAME}",
                         "KAGGLE_KEY=${params.KAGGLE_KEY}",
                         "CUTOFF=${params.CUTOFF}"]) {
                             sh './process_data.sh'
                            // sh 'python3 ./download_data_and_process.py'
                            archiveArtifacts artifacts: "data_test.csv, data_dev.csv, data_train.csv, column_titles.csv"
                }
            }
        }
    }
}
