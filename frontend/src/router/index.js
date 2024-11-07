import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import About from '../views/About.vue'
import Game from '../views/Game.vue'
import Rogue from '../views/Rogue.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
    meta: {
      title: 'Home - Game Automation Suite'
    }
  },
  {
    path: '/about',
    name: 'About',
    component: About,
    meta: {
      title: 'About - Game Automation Suite'
    }
  },
  {
    path: '/game',
    name: 'Game',
    component: Game,
    meta: {
      title: 'Game Control - Game Automation Suite'
    }
  },
  {
    path: '/rogue',
    name: 'Rogue',
    component: Rogue,
    meta: {
      title: 'Rogue Mode - Game Automation Suite'
    }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

// 全局路由守卫：更新页面标题
router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'Game Automation Suite'
  next()
})

export default router
