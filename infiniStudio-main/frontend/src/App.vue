<template>
  <a-layout style="min-height: 100vh; background: #f0f2f5">
    <a-layout-header style="position: fixed; top: 0; left: 0; right: 0; z-index: 1000; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1)">
      <div style="display: flex; align-items: center; justify-content: space-between; height: 100%">
        <div style="color: white; font-size: 24px; font-weight: bold; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 28px">⚡</span>
          <span>InfiniStudio</span>
        </div>
      </div>
    </a-layout-header>
    <a-layout>
      <a-layout-sider 
        width="220" 
        :style="{ 
          background: '#fff', 
          boxShadow: '2px 0 8px rgba(0,0,0,0.08)',
          overflow: 'auto',
          height: 'calc(100vh - 64px)',
          position: 'fixed',
          left: 0,
          top: '64px',
          zIndex: 999
        }"
      >
        <a-menu
          v-model:selectedKeys="selectedKeys"
          mode="inline"
          :style="{ 
            height: '100%', 
            borderRight: 0,
            paddingTop: '16px'
          }"
          @click="handleMenuClick"
        >
          <a-menu-item key="overview" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">📊</span>
            </template>
            <span style="font-weight: 500">总览</span>
          </a-menu-item>
          <a-menu-item key="brands" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">🏢</span>
            </template>
            <span style="font-weight: 500">品牌管理</span>
          </a-menu-item>
          <a-menu-item key="models" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">🤖</span>
            </template>
            <span style="font-weight: 500">模型管理</span>
          </a-menu-item>
          <a-menu-item key="servers" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">🖥️</span>
            </template>
            <span style="font-weight: 500">服务器管理</span>
          </a-menu-item>
          <a-menu-item key="services" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">🚀</span>
            </template>
            <span style="font-weight: 500">服务管理</span>
          </a-menu-item>
          <a-menu-item key="tasks" :style="{ marginBottom: '8px', borderRadius: '8px', marginLeft: '12px', marginRight: '12px' }">
            <template #icon>
              <span style="font-size: 18px">⏰</span>
            </template>
            <span style="font-weight: 500">计划任务</span>
          </a-menu-item>
        </a-menu>
      </a-layout-sider>
      <a-layout-content :style="{ 
        marginLeft: '220px', 
        marginTop: '64px',
        padding: '24px', 
        background: '#f0f2f5',
        minHeight: 'calc(100vh - 64px)'
      }">
        <div style="background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-height: calc(100vh - 112px)">
          <router-view />
        </div>
      </a-layout-content>
    </a-layout>
  </a-layout>
</template>

<script>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'App',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const selectedKeys = ref([route.name || 'overview'])

    const handleMenuClick = ({ key }) => {
      router.push({ name: key })
      selectedKeys.value = [key]
    }

    return {
      selectedKeys,
      handleMenuClick
    }
  }
}
</script>

<style>
#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* 美化菜单项选中状态 */
.ant-menu-item-selected {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: #fff !important;
  font-weight: 600 !important;
}

.ant-menu-item-selected .ant-menu-item-icon {
  color: #fff !important;
}

.ant-menu-item:hover {
  background: #f5f5f5 !important;
  border-radius: 8px !important;
}

.ant-menu-item-selected:hover {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

/* 美化卡片 */
.ant-card {
  border-radius: 12px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
  transition: all 0.3s ease !important;
}

.ant-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
  transform: translateY(-2px);
}

/* 美化表格 */
.ant-table {
  border-radius: 8px;
  overflow: hidden;
}

.ant-table-thead > tr > th {
  background: #fafafa !important;
  font-weight: 600 !important;
  color: #262626 !important;
}

/* 美化按钮 */
.ant-btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
}

.ant-btn-primary:hover {
  background: linear-gradient(135deg, #5568d3 0%, #653a8f 100%);
  box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
}

/* 美化统计卡片 */
.ant-statistic-title {
  color: #8c8c8c;
  font-size: 14px;
  font-weight: 500;
}

.ant-statistic-content {
  color: #262626;
  font-weight: 600;
}
</style>
