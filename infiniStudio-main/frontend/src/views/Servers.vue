<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px">
      <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
        <span style="font-size: 28px">🖥️</span>
        <span>服务器管理</span>
      </h2>
      <a-button type="primary" @click="showModal" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
        <span style="margin-right: 8px">➕</span>添加服务器
      </a-button>
    </div>

    <a-card :bordered="false">
      <a-table 
        :columns="columns" 
        :data-source="servers" 
        :pagination="false"
        row-key="id"
        :bordered="false"
      >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'brand_model'">
          {{ record.brand_name || '-' }} / {{ record.model_name || '-' }}
        </template>
        <template v-if="column.key === 'status'">
          <a-tag :color="record.status === 'online' ? 'success' : 'error'" style="font-weight: 500; padding: 4px 12px; border-radius: 4px">
            {{ record.status === 'online' ? '在线' : '离线' }}
          </a-tag>
        </template>
        <template v-if="column.key === 'action'">
          <a-button type="link" @click="editServer(record)">编辑</a-button>
          <a-button type="link" @click="openSSH(record)">SSH连接</a-button>
          <a-popconfirm title="确定删除这个服务器吗？" @confirm="deleteServer(record.id)">
            <a-button type="link" danger>删除</a-button>
          </a-popconfirm>
        </template>
      </template>
    </a-table>
    </a-card>

    <a-modal
      v-model:open="modalVisible"
      :title="editingServer ? '编辑服务器' : '添加服务器'"
      width="600px"
      @ok="handleSubmit"
      @cancel="handleCancel"
    >
      <a-form :model="form" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="服务器名称" required>
          <a-input v-model:value="form.name" placeholder="请输入服务器名称" />
        </a-form-item>
        <a-form-item label="品牌">
          <a-select v-model:value="form.brand_id" placeholder="请选择品牌" allow-clear>
            <a-select-option v-for="brand in brands" :key="brand.id" :value="brand.id">
              {{ brand.name }}
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="型号">
          <a-select 
            v-model:value="form.model_id" 
            placeholder="请选择型号" 
            :disabled="!form.brand_id"
            allow-clear
          >
            <a-select-option 
              v-for="accelerator in accelerators" 
              :key="accelerator.id" 
              :value="accelerator.id"
            >
              {{ accelerator.name }} ({{ accelerator.model }})
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="主机IP" required>
          <a-input v-model:value="form.host_ip" placeholder="请输入主机IP" />
        </a-form-item>
        <a-form-item label="端口">
          <a-input-number v-model:value="form.port" :min="1" :max="65535" style="width: 100%" />
        </a-form-item>
        <a-form-item label="用户名" required>
          <a-input v-model:value="form.username" placeholder="请输入用户名" />
        </a-form-item>
        <a-form-item label="密码">
          <a-input-password v-model:value="form.password" placeholder="请输入密码" />
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      v-model:open="sshModalVisible"
      :title="`SSH连接 - ${currentServer?.name}`"
      width="1200px"
      :footer="null"
      :maskClosable="false"
      :bodyStyle="{ padding: '0', overflow: 'hidden' }"
      @cancel="handleSSHModalClose"
    >
      <SshTerminal v-if="sshModalVisible" :server-id="currentServer?.id" :key="currentServer?.id" />
    </a-modal>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { getServers, getBrands, getAccelerators, createServer, updateServer, deleteServer as deleteServerApi, checkAllServers } from '../api'
import { message } from 'ant-design-vue'
import SshTerminal from '../components/SshTerminal.vue'

export default {
  name: 'Servers',
  components: {
    SshTerminal
  },
  setup() {
    const servers = ref([])
    const brands = ref([])
    const accelerators = ref([])
    const modalVisible = ref(false)
    const sshModalVisible = ref(false)
    const editingServer = ref(null)
    const currentServer = ref(null)
    const form = ref({
      name: '',
      brand_id: null,
      model_id: null,
      host_ip: '',
      port: 22,
      username: '',
      password: ''
    })

    const columns = [
      { title: '服务器名称', dataIndex: 'name', key: 'name' },
      { title: '品牌型号', key: 'brand_model' },
      { title: '主机IP', dataIndex: 'host_ip', key: 'host_ip' },
      { title: '状态', key: 'status' },
      { title: '操作', key: 'action' }
    ]

    const loadData = async () => {
      try {
        const [serversRes, brandsRes] = await Promise.all([
          getServers(),
          getBrands()
        ])
        servers.value = serversRes.data
        brands.value = brandsRes.data
      } catch (error) {
        message.error('加载数据失败')
      }
    }

    const checkServerStatus = async () => {
      try {
        const res = await checkAllServers()
        // 更新服务器状态
        const statusMap = {}
        res.data.results.forEach(result => {
          statusMap[result.id] = result.status
        })
        servers.value = servers.value.map(server => ({
          ...server,
          status: statusMap[server.id] || server.status
        }))
      } catch (error) {
        console.error('检查服务器状态失败:', error)
      }
    }

    watch(() => form.value.brand_id, async (brandId) => {
      if (brandId) {
        try {
          const res = await getAccelerators(brandId)
          accelerators.value = res.data
        } catch (error) {
          accelerators.value = []
        }
      } else {
        accelerators.value = []
        form.value.model_id = null
      }
    })

    const showModal = () => {
      editingServer.value = null
      form.value = {
        name: '',
        brand_id: null,
        model_id: null,
        host_ip: '',
        port: 22,
        username: '',
        password: ''
      }
      accelerators.value = []
      modalVisible.value = true
    }

    const editServer = async (server) => {
      editingServer.value = server
      form.value = {
        name: server.name,
        brand_id: server.brand_id,
        model_id: server.model_id,
        host_ip: server.host_ip,
        port: server.port,
        username: server.username,
        password: server.password || ''
      }
      if (server.brand_id) {
        try {
          const res = await getAccelerators(server.brand_id)
          accelerators.value = res.data
        } catch (error) {
          accelerators.value = []
        }
      }
      modalVisible.value = true
    }

    const handleSubmit = async () => {
      if (!form.value.name || !form.value.host_ip || !form.value.username) {
        message.warning('请填写必填项')
        return
      }
      
      try {
        if (editingServer.value) {
          await updateServer(editingServer.value.id, form.value)
          message.success('更新成功')
        } else {
          await createServer(form.value)
          message.success('创建成功')
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

    const deleteServer = async (id) => {
      try {
        await deleteServerApi(id)
        message.success('删除成功')
        loadData()
      } catch (error) {
        message.error('删除失败')
      }
    }

    const openSSH = (server) => {
      currentServer.value = server
      sshModalVisible.value = true
    }

    const handleSSHModalClose = () => {
      // 模态框关闭时会触发组件卸载，组件会自己处理连接清理
      sshModalVisible.value = false
      currentServer.value = null
    }

    let statusCheckInterval = null

    onMounted(() => {
      loadData()
      // 每30秒检查一次服务器状态
      statusCheckInterval = setInterval(checkServerStatus, 30000)
      // 立即检查一次
      checkServerStatus()
    })

    onUnmounted(() => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval)
      }
    })

    return {
      servers,
      brands,
      accelerators,
      columns,
      modalVisible,
      sshModalVisible,
      editingServer,
      currentServer,
      form,
      showModal,
      editServer,
      handleSubmit,
      handleCancel,
      deleteServer,
      openSSH,
      handleSSHModalClose
    }
  }
}
</script>

