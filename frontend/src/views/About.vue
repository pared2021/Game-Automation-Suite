<template>
  <div class="about">
    <h1>{{ $t('about.title') }}</h1>

    <div class="content-grid">
      <!-- 项目介绍 -->
      <section class="info-card">
        <h2>{{ $t('about.projectInfo') }}</h2>
        <p>{{ $t('about.description') }}</p>
        <div class="version-info">
          <span>{{ $t('about.version') }}: v1.0.0</span>
        </div>
      </section>

      <!-- 功能特性 -->
      <section class="info-card features">
        <h2>{{ $t('about.features') }}</h2>
        <ul class="feature-list">
          <li>
            <i class="feature-icon">🎮</i>
            <div class="feature-text">
              <h3>{{ $t('about.gameAutomation') }}</h3>
              <p>{{ $t('about.gameAutomationDesc') }}</p>
            </div>
          </li>
          <li>
            <i class="feature-icon">🤖</i>
            <div class="feature-text">
              <h3>{{ $t('about.aiDecision') }}</h3>
              <p>{{ $t('about.aiDecisionDesc') }}</p>
            </div>
          </li>
          <li>
            <i class="feature-icon">🎲</i>
            <div class="feature-text">
              <h3>{{ $t('about.rogueMode') }}</h3>
              <p>{{ $t('about.rogueModeDesc') }}</p>
            </div>
          </li>
          <li>
            <i class="feature-icon">📊</i>
            <div class="feature-text">
              <h3>{{ $t('about.dataAnalysis') }}</h3>
              <p>{{ $t('about.dataAnalysisDesc') }}</p>
            </div>
          </li>
        </ul>
      </section>

      <!-- 技术栈 -->
      <section class="info-card tech-stack">
        <h2>{{ $t('about.techStack') }}</h2>
        <div class="tech-grid">
          <div class="tech-item">
            <img src="@/assets/vue.png" alt="Vue.js">
            <span>Vue.js</span>
          </div>
          <div class="tech-item">
            <img src="@/assets/python.png" alt="Python">
            <span>Python</span>
          </div>
          <div class="tech-item">
            <img src="@/assets/tensorflow.png" alt="TensorFlow">
            <span>TensorFlow</span>
          </div>
          <div class="tech-item">
            <img src="@/assets/opencv.png" alt="OpenCV">
            <span>OpenCV</span>
          </div>
        </div>
      </section>

      <!-- 贡献者 -->
      <section class="info-card contributors">
        <h2>{{ $t('about.contributors') }}</h2>
        <div class="contributor-list">
          <div class="contributor">
            <img src="@/assets/avatar1.png" alt="Contributor 1">
            <h3>Developer 1</h3>
            <p>Lead Developer</p>
          </div>
          <div class="contributor">
            <img src="@/assets/avatar2.png" alt="Contributor 2">
            <h3>Developer 2</h3>
            <p>AI Engineer</p>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script>
export default {
  name: 'About'
}
</script>

<style scoped>
.about {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.content-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.info-card {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.version-info {
  margin-top: 15px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
  text-align: center;
}

.feature-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.feature-list li {
  display: flex;
  align-items: flex-start;
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.3s ease;
}

.feature-list li:hover {
  transform: translateY(-2px);
}

.feature-icon {
  font-size: 24px;
  margin-right: 15px;
  background: white;
  padding: 10px;
  border-radius: 50%;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.feature-text {
  flex: 1;
}

.feature-text h3 {
  margin: 0 0 5px 0;
  color: #2c3e50;
}

.feature-text p {
  margin: 0;
  color: #666;
}

.tech-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.tech-item {
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.3s ease;
}

.tech-item:hover {
  transform: translateY(-2px);
}

.tech-item img {
  width: 50px;
  height: 50px;
  margin-bottom: 10px;
}

.contributor-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 20px;
  margin-top: 15px;
}

.contributor {
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.3s ease;
}

.contributor:hover {
  transform: translateY(-2px);
}

.contributor img {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  margin-bottom: 10px;
}

.contributor h3 {
  margin: 0 0 5px 0;
  color: #2c3e50;
}

.contributor p {
  margin: 0;
  color: #666;
}

@media (max-width: 768px) {
  .content-grid {
    grid-template-columns: 1fr;
  }

  .tech-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .contributor-list {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
