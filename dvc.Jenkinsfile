pipeline {
  	agent {
		dockerfile true
	}
        parameters {
            password(
                defaultValue: '',
                description: 'DVC pass',
                name: 'IUM_SFTP_KEY'
            )
    }
    stages {
        stage('checkout: Check out from version control') {
            steps { 
                    checkout([$class: 'GitSCM', branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('reproduce') {
            steps {
                withCredentials(
                    [sshUserPrivateKey(credentialsId: '48ac7004-216e-4260-abba-1fe5db753e18', keyFileVariable: 'IUM_SFTP_KEY', )]) {
                                sh 'dvc remote add -f -d ium_ssh_remote ssh://ium-sftp@tzietkiewicz.vm.wmi.amu.edu.pl/ium-sftp'
                                sh 'dvc remote modify --local ium_ssh_remote keyfile $IUM_SFTP_KEY'
                                sh 'dvc pull'
                                sh 'dvc repro'
                                sh ll
                                }
                }
            }
        }
}
