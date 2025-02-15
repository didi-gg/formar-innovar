Sub CrearVistaEstudiantes()
    Dim wsBase As Worksheet
    Dim wsView As Worksheet
    Dim lastRow As Long
    Dim currentRow As Long
    Dim currentLevel As String, currentGrade As String
    Dim gradeCollection As Collection
    Dim key As Variant
    Dim i As Long
    Dim activeCount As Long
    
    ' Configurar hoja base
    Set wsBase = ThisWorkbook.Sheets("BD")
    
    ' Eliminar la hoja "Vista-Estudiantes" si ya existe
    On Error Resume Next
    Application.DisplayAlerts = False
    Set wsView = ThisWorkbook.Sheets("Vista-Estudiantes")
    If Not wsView Is Nothing Then
        wsView.Delete
    End If
    Application.DisplayAlerts = True
    On Error GoTo 0
    
    ' Crear la hoja "Vista-Estudiantes"
    Set wsView = ThisWorkbook.Sheets.Add
    wsView.Name = "Vista-Estudiantes"
    
    ' Mover la hoja "Vista-Estudiantes" a la segunda posición
    wsView.Move After:=ThisWorkbook.Sheets("BD")
    
    ' Encabezado general
    With wsView.Range("A1:R1")
        .Merge ' Combinar celdas
        .Value = "Lista de estudiantes " & Year(Date)
        .Font.Name = "Calibri (Body)" ' Fuente Calibri
        .Font.Size = 16 ' Tamaño 16
        .Font.Bold = False ' Sin negrita
        .Font.Color = RGB(255, 255, 255) ' Texto blanco
        .HorizontalAlignment = xlCenter ' Centrado horizontal
        .VerticalAlignment = xlCenter ' Centrado vertical
        .Interior.Color = RGB(32, 78, 120) ' Fondo 204E78
        .RowHeight = 25 ' Altura de fila
    End With

    ' Crear colección para niveles y grados únicos
    Set gradeCollection = New Collection
    lastRow = wsBase.Cells(wsBase.Rows.Count, "A").End(xlUp).Row

    ' Crear lista de niveles y grados únicos
    On Error Resume Next
    For i = 2 To lastRow
        currentLevel = wsBase.Cells(i, 1).Value ' Nivel
        currentGrade = wsBase.Cells(i, 2).Value ' Grado
        ' Combinar nivel y grado como clave única
        key = currentLevel & "|" & currentGrade
        ' Agregar a la colección si no existe
        gradeCollection.Add key, key
    Next i
    On Error GoTo 0

    ' Generar tablas por nivel y grado
    currentRow = 3
    For Each key In gradeCollection
        currentLevel = Split(key, "|")(0)
        currentGrade = Split(key, "|")(1)

        ' Inicializar conteo de estudiantes activos
        activeCount = 0

        ' Calcular conteo de estudiantes activos y copiar datos relevantes
        For i = 2 To lastRow
            If wsBase.Cells(i, 1).Value = currentLevel And wsBase.Cells(i, 2).Value = currentGrade Then
                If wsBase.Cells(i, 4).Value = "Activo" Then ' Estado = Activo
                    activeCount = activeCount + 1
                End If
            End If
        Next i

        ' Mostrar conteo de estudiantes activos
        With wsView.Range(wsView.Cells(currentRow, 1), wsView.Cells(currentRow, 18)) ' Rango A:R
            .Merge ' Combinar celdas
            .Value = "Número de estudiantes activos: " & activeCount
            .HorizontalAlignment = xlLeft ' Alineado a la izquierda
            .VerticalAlignment = xlCenter
            .Font.Bold = False ' Sin negrita
            .Interior.Color = RGB(222, 234, 247) ' Fondo DEEAF7
            .Borders.LineStyle = xlContinuous ' Borde sencillo
            .Borders.Weight = xlThin
        End With
        currentRow = currentRow + 1

        ' Encabezado del grupo
        With wsView.Range(wsView.Cells(currentRow, 1), wsView.Cells(currentRow, 18)) ' Rango A:R
            .Merge ' Combinar celdas
            .Value = currentLevel & " - " & currentGrade
            .HorizontalAlignment = xlCenter ' Centrado horizontalmente
            .VerticalAlignment = xlCenter ' Centrado verticalmente
            .Font.Bold = False ' Sin negrita
            .Interior.Color = RGB(0, 112, 192) ' Fondo azul (0, 112, 192)
            .Font.Color = RGB(255, 255, 255) ' Texto blanco
            .Borders.LineStyle = xlContinuous ' Borde sencillo
            .Borders.Weight = xlThin
        End With
        currentRow = currentRow + 1

        ' Copiar encabezados, ajustados al rango destino (A:R)
        With wsView.Range(wsView.Cells(currentRow, 1), wsView.Cells(currentRow, 18)) ' Rango A:R
            ' Copiar encabezados desde la hoja base
            wsBase.Range(wsBase.Cells(1, 3), wsBase.Cells(1, 20)).Copy Destination:=.Cells(1, 1)
            ' Aplicar formato
            .Font.Bold = True ' Texto en negrita
            .Font.Color = RGB(0, 0, 0) ' Texto negro
            .HorizontalAlignment = xlCenter ' Centrado
            .Interior.Color = RGB(155, 194, 230) ' Fondo 9BC2E6
            .Borders.LineStyle = xlContinuous ' Borde sencillo
            .Borders.Weight = xlThin
        End With
        currentRow = currentRow + 1

        ' Copiar datos relevantes
        For i = 2 To lastRow
            If wsBase.Cells(i, 1).Value = currentLevel And wsBase.Cells(i, 2).Value = currentGrade Then
                ' Copiar los datos de la columna C a la T en el rango A:R
                wsBase.Range(wsBase.Cells(i, 3), wsBase.Cells(i, 20)).Copy _
                    Destination:=wsView.Cells(currentRow, 1)
                
                ' Aplicar borde sencillo al rango copiado
                With wsView.Range(wsView.Cells(currentRow, 1), wsView.Cells(currentRow, 18)) ' Hasta la columna R
                    .Borders.LineStyle = xlContinuous
                    .Borders.Weight = xlThin
                End With
                
                ' Incrementar la fila actual
                currentRow = currentRow + 1
            End If
        Next i
        currentRow = currentRow + 1
        
    Next key

    ' Ajustar formato
    wsView.Columns.AutoFit

    ' Fijar la vista desde la columna L
    wsView.Cells(2, 12).Select ' Columna L (12ª columna)
    ActiveWindow.FreezePanes = True

    ' Proteger hoja como solo lectura
    wsView.Protect Password:="12345", AllowSorting:=True, AllowFiltering:=True, AllowUsingPivotTables:=False

    MsgBox "Vista de estudiantes creada y protegida con éxito.", vbInformation
End Sub
