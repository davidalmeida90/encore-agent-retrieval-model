import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { ChatLayout } from '@/components/chat/ChatLayout'
import { LandingPage } from '@/pages/LandingPage'
import { ProtectedRoute } from '@/components/ProtectedRoute'
import { PublicRoute } from '@/components/PublicRoute'
import { ChatEmptyPage } from '@/pages/chat/ChatEmptyPage'
import { ChatThreadPage } from '@/pages/chat/ChatThreadPage'
import { Login } from '@/pages/Login'
import { SignUp } from '@/pages/SignUp'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        <Route
          path="/signup"
          element={
            <PublicRoute>
              <SignUp />
            </PublicRoute>
          }
        />
        <Route
          path="/"
          element={
            <PublicRoute>
              <LandingPage />
            </PublicRoute>
          }
        />
        <Route
          path="/chats"
          element={
            <ProtectedRoute>
              <ChatLayout />
            </ProtectedRoute>
          }
        >
          <Route index element={<ChatEmptyPage />} />
          <Route path=":threadId" element={<ChatThreadPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/chats" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
