pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build & Test') {
            steps {
                sh 'mvn -B test || true'
            }
        }

        stage('Generate Allure Report') {
            steps {
                sh 'mvn -B allure:report'
            }
        }

        stage('Publish') {
            steps {
                allure includeProperties: false, results: [[path: 'target/allure-results']]
                publishHTML(target: [
                    reportName: 'Allure Report',
                    reportDir: 'target/site/allure-maven-plugin',
                    reportFiles: 'index.html',
                    keepAll: true,
                    alwaysLinkToLastBuild: true
                ])
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
    }
}
