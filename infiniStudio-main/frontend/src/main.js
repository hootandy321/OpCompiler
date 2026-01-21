import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'

// 静默 ResizeObserver 循环警告（这是浏览器的已知警告，不影响功能）
const resizeObserverErrorHandler = (e) => {
  if (e.message && e.message.includes('ResizeObserver loop')) {
    e.stopImmediatePropagation()
    return false
  }
}
window.addEventListener('error', resizeObserverErrorHandler, true)

const app = createApp(App)
app.use(router)
app.use(Antd)
app.mount('#app')

