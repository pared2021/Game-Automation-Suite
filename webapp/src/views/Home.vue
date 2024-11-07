<template>
  <div class="home">
    <h1>Welcome to Game Automation</h1>
    <p>This project aims to automate various aspects of game playing using advanced AI and image recognition techniques.</p>
    <div class="features">
      <h2>Features:</h2>
      <ul>
        <li>Automated battle strategies</li>
        <li>Rogue-like game mode</li>
        <li>Advanced blessing system</li>
        <li>AI-driven decision making</li>
      </ul>
    </div>
    <router-link to="/rogue" class="cta-button">Try Rogue Mode</router-link>
  </div>
</template>

<script>
export default {
  name: 'Home'
}
</script>

<style scoped>
.home {
  text-align: left;
}

.features {
  margin-top: 20px;
  margin-bottom: 20px;
}

.cta-button {
  display: inline-block;
  background-color: #42b983;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  text-decoration: none;
  font-weight: bold;
  transition: background-color 0.3s;
}

.cta-button:hover {
  background-color: #3aa876;
}
</style>