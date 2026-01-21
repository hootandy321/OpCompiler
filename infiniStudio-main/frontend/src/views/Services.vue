<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px">
      <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
        <span style="font-size: 28px">🚀</span>
        <span>服务管理</span>
      </h2>
      <div style="display: flex; gap: 12px; align-items: center">
        <a-button @click="refreshStatus" :loading="refreshing" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
          <span style="margin-right: 8px">🔄</span>刷新状态
        </a-button>
        <a-button type="primary" @click="showModal" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
          <span style="margin-right: 8px">➕</span>部署服务
        </a-button>
      </div>
    </div>

    <a-card :bordered="false">
      <a-table 
        :columns="columns" 
        :data-source="services" 
        :pagination="false"
        row-key="id"
        :bordered="false"
      >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'servers'">
          <a-tag color="blue" style="font-weight: 500">
            {{ record.server_ids?.length || 0 }} 个服务器
          </a-tag>
        </template>
        <template v-if="column.key === 'deploy_status'">
          <a-tag 
            :color="getDeployStatusColor(record.deploy_status)" 
            style="font-weight: 500; cursor: pointer"
            @click="viewDeployLog(record)"
          >
            {{ getDeployStatusText(record.deploy_status) }}
          </a-tag>
        </template>
        <template v-if="column.key === 'action'">
          <a-button type="link" @click="editService(record)" style="color: #667eea; font-weight: 500; padding: 0 8px">编辑</a-button>
          <a-button type="link" @click="restartService(record)" style="color: #faad14; font-weight: 500; padding: 0 8px">重启服务</a-button>
          <a-button type="link" @click="stopService(record)" style="color: #ff4d4f; font-weight: 500; padding: 0 8px">停止服务</a-button>
          <a-button type="link" @click="enterService(record)" style="color: #52c41a; font-weight: 500; padding: 0 8px">进入服务</a-button>
          <a-popconfirm title="确定删除这个服务吗？" @confirm="deleteService(record.id)">
            <a-button type="link" danger style="font-weight: 500; padding: 0 8px">删除</a-button>
          </a-popconfirm>
        </template>
      </template>
    </a-table>
    </a-card>

    <a-modal
      v-model:open="modalVisible"
      :title="editingService ? '编辑服务' : '部署服务'"
      width="600px"
      @ok="handleSubmit"
      @cancel="handleCancel"
    >
      <a-form :model="form" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="服务名称" required>
          <a-input v-model:value="form.name" placeholder="请输入服务名称" />
        </a-form-item>
        <a-form-item label="模型" required>
          <a-select v-model:value="form.model_id" placeholder="请选择模型">
            <a-select-option v-for="model in models" :key="model.id" :value="model.id">
              {{ model.name }}
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="服务器" required>
          <a-select 
            v-model:value="form.server_ids" 
            mode="multiple" 
            placeholder="请选择服务器"
          >
            <a-select-option v-for="server in servers" :key="server.id" :value="server.id">
              {{ server.name }} ({{ server.host_ip }})
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="部署命令" required>
          <a-textarea 
            v-model:value="form.deploy_command" 
            placeholder="请输入部署命令，例如：docker run -d -p 8080:8080 my-service"
            :rows="4"
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      v-model:open="deployLogVisible"
      title="部署日志"
      width="900px"
      :footer="null"
    >
      <div v-if="deployLogs.length === 0" style="text-align: center; padding: 40px; color: #8c8c8c">
        暂无部署日志
      </div>
      <div v-else>
        <div v-for="(log, index) in deployLogs" :key="index" style="margin-bottom: 24px">
          <a-card :bordered="false" style="background: #fafafa">
            <template #title>
              <div style="display: flex; justify-content: space-between; align-items: center">
                <span style="font-weight: 600">
                  {{ log.server_name }} ({{ log.server_ip }})
                </span>
                <a-tag :color="log.success ? 'success' : 'error'">
                  {{ log.success ? '成功' : '失败' }}
                </a-tag>
              </div>
            </template>
            <div style="margin-top: 12px">
              <div v-if="log.output" style="margin-bottom: 12px">
                <div style="font-weight: 500; margin-bottom: 8px; color: #262626">标准输出：</div>
                <pre style="background: #fff; padding: 12px; border-radius: 4px; border: 1px solid #e8e8e8; margin: 0; white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto">{{ log.output }}</pre>
              </div>
              <div v-if="log.error">
                <div style="font-weight: 500; margin-bottom: 8px; color: #ff4d4f">错误输出：</div>
                <pre style="background: #fff2f0; padding: 12px; border-radius: 4px; border: 1px solid #ffccc7; margin: 0; white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto; color: #cf1322">{{ log.error }}</pre>
              </div>
            </div>
          </a-card>
        </div>
      </div>
    </a-modal>

  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getServices, refreshServicesStatus, getModels, getServers, createService, updateService, deleteService as deleteServiceApi, restartService as restartServiceApi, stopService as stopServiceApi, getDeployLog } from '../api'
import { message } from 'ant-design-vue'

export default {
  name: 'Services',
  setup() {
    const router = useRouter()
    const services = ref([])
    const models = ref([])
    const servers = ref([])
    const modalVisible = ref(false)
    const editingService = ref(null)
    const deployLogVisible = ref(false)
    const deployLogs = ref([])
    const shownErrors = ref(new Set())  // 记录已显示的错误，避免重复弹出
    const refreshing = ref(false)  // 刷新状态加载中
    const form = ref({
      name: '',
      model_id: null,
      server_ids: [],
      deploy_command: ''
    })

    const columns = [
      { title: '服务名称', dataIndex: 'name', key: 'name' },
      { title: '模型名称', dataIndex: 'model_name', key: 'model_name' },
      { title: '服务器', key: 'servers' },
      { title: '部署状态', key: 'deploy_status' },
      { title: '操作', key: 'action' }
    ]
    
    const getDeployStatusColor = (status) => {
      const colors = {
        'offline': 'default',
        'online': 'blue',
        'deploying': 'processing',
        'deployed': 'success'
      }
      return colors[status] || 'default'
    }
    
    const getDeployStatusText = (status) => {
      const texts = {
        'offline': '离线',
        'online': '在线',
        'deploying': '部署中',
        'deployed': '已部署'
      }
      return texts[status] || status
    }

    const refreshStatus = async () => {
      refreshing.value = true
      try {
        const servicesRes = await refreshServicesStatus()
        services.value = servicesRes.data
        message.success('状态刷新成功')
        
        // 检查是否有部署失败的服务，弹出错误信息
        services.value.forEach(service => {
          if (service.deploy_result) {
            try {
              const results = JSON.parse(service.deploy_result)
              if (Array.isArray(results)) {
                const failedServers = results.filter(r => !r.success)
                if (failedServers.length > 0) {
                  // 构建错误信息的唯一标识（使用服务ID和错误内容的hash）
                  const errorKey = `${service.id}_${JSON.stringify(results)}`
                  
                  // 只在第一次检测到错误时弹出
                  if (!shownErrors.value.has(errorKey)) {
                    shownErrors.value.add(errorKey)
                    
                    // 构建错误信息
                    const errorMessages = failedServers.map(s => {
                      let msg = `${s.server_name || s.server_ip}: `
                      if (s.error) {
                        msg += s.error
                      } else if (s.message) {
                        msg += s.message
                      } else {
                        msg += '部署失败'
                      }
                      return msg
                    }).join('\n')
                    
                    message.error({
                      content: `服务 "${service.name}" 部署失败：\n${errorMessages}`,
                      duration: 8
                    })
                  }
                }
              }
            } catch (e) {
              // 解析失败，忽略
            }
          }
        })
      } catch (error) {
        message.error('刷新状态失败')
      } finally {
        refreshing.value = false
      }
    }

    const loadData = async () => {
      try {
        const [servicesRes, modelsRes, serversRes] = await Promise.all([
          getServices(),
          getModels(),
          getServers()
        ])
        services.value = servicesRes.data
        models.value = modelsRes.data
        servers.value = serversRes.data
        
        // 检查是否有部署失败的服务，弹出错误信息
        services.value.forEach(service => {
          if (service.deploy_result) {
            try {
              const results = JSON.parse(service.deploy_result)
              if (Array.isArray(results)) {
                const failedServers = results.filter(r => !r.success)
                if (failedServers.length > 0) {
                  // 构建错误信息的唯一标识（使用服务ID和错误内容的hash）
                  const errorKey = `${service.id}_${JSON.stringify(results)}`
                  
                  // 只在第一次检测到错误时弹出
                  if (!shownErrors.value.has(errorKey)) {
                    shownErrors.value.add(errorKey)
                    
                    // 构建错误信息
                    const errorMessages = failedServers.map(s => {
                      let msg = `${s.server_name || s.server_ip}: `
                      if (s.error) {
                        msg += s.error
                      } else if (s.message) {
                        msg += s.message
                      } else {
                        msg += '部署失败'
                      }
                      return msg
                    }).join('\n')
                    
                    message.error({
                      content: `服务 "${service.name}" 部署失败：\n${errorMessages}`,
                      duration: 8
                    })
                  }
                }
              }
            } catch (e) {
              // 解析失败，忽略
            }
          }
        })
      } catch (error) {
        message.error('加载数据失败')
      }
    }

    const showModal = () => {
      editingService.value = null
      form.value = {
        name: '',
        model_id: null,
        server_ids: [],
        deploy_command: ''
      }
      modalVisible.value = true
    }

    const editService = (service) => {
      editingService.value = service
      form.value = {
        name: service.name,
        model_id: service.model_id,
        server_ids: service.server_ids || [],
        deploy_command: service.deploy_command || ''
      }
      modalVisible.value = true
    }

    const handleSubmit = async () => {
      if (!form.value.name || !form.value.model_id || !form.value.server_ids?.length || !form.value.deploy_command) {
        message.warning('请填写必填项')
        return
      }
      
      try {
        let serviceId
        if (editingService.value) {
          await updateService(editingService.value.id, form.value)
          message.success('更新成功')
          serviceId = editingService.value.id
        } else {
          const res = await createService(form.value)
          message.success('创建成功')
          serviceId = res.data.id
        }
        modalVisible.value = false
        loadData()
      } catch (error) {
        message.error('操作失败')
      }
    }

    const handleCancel = () => {
      modalVisible.value = false
    }

    const deleteService = async (id) => {
      try {
        await deleteServiceApi(id)
        message.success('删除成功')
        loadData()
      } catch (error) {
        message.error('删除失败')
      }
    }

    const restartService = async (service) => {
      try {
        await restartServiceApi(service.id)
        message.success('重启服务已启动')
        // 延迟一下再刷新数据，以便显示状态变化
        setTimeout(() => {
          loadData()
        }, 1000)
      } catch (error) {
        message.error('重启服务失败')
      }
    }

    const stopService = async (service) => {
      try {
        await stopServiceApi(service.id)
        message.success('停止服务已启动')
        // 延迟一下再刷新数据，以便显示状态变化
        setTimeout(() => {
          loadData()
        }, 1000)
      } catch (error) {
        message.error('停止服务失败')
      }
    }

    const viewDeployLog = async (service) => {
      // 点击部署状态，查看部署日志
      try {
        const res = await getDeployLog(service.id)
        deployLogs.value = res.data
        deployLogVisible.value = true
      } catch (error) {
        message.error('获取部署日志失败')
      }
    }

    const enterService = (service) => {
      router.push({ name: 'serviceChat', params: { id: service.id } })
    }

    onMounted(() => {
      loadData()
    })

    return {
      services,
      models,
      servers,
      columns,
      modalVisible,
      editingService,
      form,
      refreshing,
      showModal,
      editService,
      handleSubmit,
      handleCancel,
      deleteService,
      restartService,
      stopService,
      refreshStatus,
      viewDeployLog,
      enterService,
      getDeployStatusColor,
      getDeployStatusText,
      deployLogVisible,
      deployLogs
    }
  }
}
</script>

