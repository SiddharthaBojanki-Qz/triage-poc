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

        stage('Triage') {
            steps {
                withCredentials([string(credentialsId: 'anthropic-api-key', variable: 'ANTHROPIC_API_KEY')]) {
                    sh '''
                        pip install --quiet --break-system-packages anthropic
                        python3 scripts/triage.py \
                            --report-dir target/allure-results \
                            --build-url $BUILD_URL \
                            --output triage-report.md
                    '''
                }
            }
        }

        stage('Publish') {
            steps {
                allure includeProperties: false, results: [[path: 'target/allure-results']]
                archiveArtifacts artifacts: 'triage-report.md', fingerprint: true
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
    }
}
