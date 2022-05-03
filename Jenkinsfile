pipeline {
  	agent {
		dockerfile true
	}
    parameters {
        string (
            defaultValue: '10',
            description: 'Epochs number',
            name: 'EPOCH',
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
                withEnv(["EPOCH=${params.CUTOFF}"]) {
                            copyArtifacts filter: '*', projectName: 's444463-create-dataset'
                            sh 'python3 ./deepl.py $EPOCH'
                            archiveArtifacts artifacts: "model"
                }
            }
        }
    }
}
